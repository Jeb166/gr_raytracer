#!/usr/bin/env python3
"""Real-time Schwarzschild null-geodesic ray tracer (CUDA via CuPy)."""
from __future__ import annotations

import argparse
import dataclasses
import math
import struct
import time
import zlib

import numpy as np

import gr_physics as phys

__version__ = "0.7.0"

COLOR_MODES = ("gray", "blackbody", "classic")
ORDER_LIMITS = ((0, "PRIMARY"), (1, "PRIMARY+SECONDARY"), (11, "ALL"))

# Geodesic tables are built once, in double precision, and stored as float32.
TABLE_CUDA = r"""
#define B_CRIT 5.196152422706632

__device__ inline void rk4(double& u,double& v,const double h){
 const double k1u=v,k1v=3.0*u*u-u;
 const double u2=u+0.5*h*k1u,v2=v+0.5*h*k1v,k2u=v2,k2v=3.0*u2*u2-u2;
 const double u3=u+0.5*h*k2u,v3=v+0.5*h*k2v,k3u=v3,k3v=3.0*u3*u3-u3;
 const double u4=u+h*k3u,v4=v+h*k3v,k4u=v4,k4v=3.0*u4*u4-u4;
 u+=(h/6.0)*(k1u+2.0*k2u+2.0*k3u+k4u);
 v+=(h/6.0)*(k1v+2.0*k2v+2.0*k3v+k4v);
}

// Sky transfer: for each local ray cosine mu in [-1,0], the orbital angle at
// which the ray leaves through the background sphere (u_bg = 0: infinity).
extern "C" __global__
void build_transfer(unsigned char* escaped,float* phi_exit,const int n,
 const double r_cam,const double u_bg,const double step,const int max_steps){
 const int i=blockDim.x*blockIdx.x+threadIdx.x; if(i>=n)return;
 escaped[i]=0; phi_exit[i]=0.0f;
 const double mu=-1.0+(double)i/(double)(n-1);
 const double tr=sqrt(fmax(0.0,1.0-mu*mu));
 const double b=r_cam*tr/sqrt(1.0-2.0/r_cam);
 if(b<=B_CRIT)return;
 double u=1.0/r_cam;
 double v=sqrt(fmax(0.0,1.0/(b*b)-u*u+2.0*u*u*u));
 for(int it=0;it<max_steps;++it){
  double nu=u,nv=v; rk4(nu,nv,step);
  if(!isfinite(nu)||nu>=0.5)return;
  if(nu<=u_bg&&nv<0.0){
   // Interpolate the crossing inside the step; rounding it to a whole
   // step would shear the lensed sky into visible stairs.
   escaped[i]=1;
   phi_exit[i]=(float)(((double)it+(u-u_bg)/(u-nu))*step);
   return;
  }
  u=nu; v=nv;
 }
}

// Path cache: u(phi)=1/r along each ray, plus the angle of its first crossing
// of the outer disk radius.
extern "C" __global__
void build_path_table(float* tab,unsigned char* rout_hit,float* phi_rout,
 const int nmu,const int nphi,const int store_every,
 const double r_cam,const double disk_rout,const double step){
 const int i=blockDim.x*blockIdx.x+threadIdx.x; if(i>=nmu)return;
 const double mu=-1.0+(double)i/(double)(nmu-1);
 const double tr=sqrt(fmax(0.0,1.0-mu*mu));
 float* row=tab+(long long)i*nphi;
 double u=1.0/r_cam;
 row[0]=(float)u; rout_hit[i]=0; phi_rout[i]=0.0f;
 const double uout=1.0/disk_rout;

 // The exactly radial ray has L=0, so phi is not a usable parameter, but it
 // reaches every radius: its outer-radius crossing is the phi -> 0 limit.
 if(tr<1.0e-9){rout_hit[i]=1;return;}

 const double b=r_cam*tr/sqrt(1.0-2.0/r_cam);
 double v=sqrt(fmax(0.0,1.0/(b*b)-u*u+2.0*u*u*u));
 int s=1;
 const int maxit=(nphi-1)*store_every;
 for(int it=1;it<=maxit;++it){
  double nu=u,nv=v; rk4(nu,nv,step);
  if(!isfinite(nu)||nu>=0.5)return;
  if(!rout_hit[i]&&u<uout&&nu>=uout){
   rout_hit[i]=1;
   phi_rout[i]=(float)(((double)(it-1)+(uout-u)/(nu-u))*step);
  }
  u=nu; v=nv;
  if(u<0.5*uout&&v<0.0)return;  // receding beyond 2 r_out: never comes back
  if(it%store_every==0){row[s]=(float)u;++s;}
 }
}
"""

RENDER_CUDA = r"""
#define PI 3.14159265358979323846f
#define B_CRIT 5.196152422706632f
#define LIMB_K 2.06f

struct Scene{
 const unsigned char* sky; int sw,sh; const float* srgb_lut;
 const unsigned char* esc; const float* pexit; int tn;
 const float* path; const unsigned char* rout_hit; const float* phi_rout;
 int nmu,nphi; float dphi,phimax;
 const float* flux_lut; int nflux;
 const float* bb_lut; int nbb; float bb_lnT0,bb_inv_dlnT,bb_lnY_ref;
 float er[3],fw[3],rt[3],up[3];
 float rcam,fcam,rin,rout,disk_sin_half,rotation_sign,tmax,exposure;
 int disk_on,grid_on,rel_on,limb_on,order_debug,g_debug,color_mode,max_order;
};

__device__ inline float clampf(float x,float lo,float hi){return fminf(hi,fmaxf(lo,x));}

__device__ inline float srgb_oetf(float c){
 c=clampf(c,0.0f,1.0f);
 return (c<=0.0031308f)?12.92f*c:1.055f*powf(c,1.0f/2.4f)-0.055f;
}

// Bilinear lookup of u=1/r in the (mu, phi) path cache; 0 means "no data"
// (the ray was captured or left the disk region before this angle).
__device__ float path_u(const Scene& S,float mu,float ph){
 if(ph<0.0f)return 0.0f;
 const int nmu=S.nmu,nphi=S.nphi;
 float jp=ph/S.dphi;
 int j0=(int)floorf(jp);
 if(j0<0||j0>=nphi-1)return 0.0f;
 int j1=j0+1;
 float fp=jp-j0;

 float mp=clampf((mu+1.0f)*(nmu-1),0.0f,(float)(nmu-1));
 int i0=(int)floorf(mp),i1=min(i0+1,nmu-1);
 float fm=mp-i0;

 const float* p=S.path;
 long long a0=(long long)i0*nphi,a1=(long long)i1*nphi;
 float a=p[a0+j0],b=p[a0+j1],c=p[a1+j0],d=p[a1+j1];
 if(a>0.0f&&b>0.0f&&c>0.0f&&d>0.0f){
  float x=a*(1.0f-fp)+b*fp;
  float y=c*(1.0f-fp)+d*fp;
  return x*(1.0f-fm)+y*fm;
 }

 // phi is singular for the exactly radial L=0 ray; borrow its neighbour.
 if(i0==0&&nmu>1){
  float x=p[nphi+j0],y=p[nphi+j1];
  if(x>0.0f&&y>0.0f)return x*(1.0f-fp)+y*fp;
 }

 long long q=(long long)((fm<0.5f)?i0:i1)*nphi;
 a=p[q+j0]; b=p[q+j1];
 return (a>0.0f&&b>0.0f)?a*(1.0f-fp)+b*fp:0.0f;
}

// Angle at which the ray first crosses the outer disk radius, if it does.
__device__ int rout_crossing(const Scene& S,float mu,float* phit){
 float mp=clampf((mu+1.0f)*(S.nmu-1),0.0f,(float)(S.nmu-1));
 int i0=(int)floorf(mp),i1=min(i0+1,S.nmu-1);
 float f=mp-i0;
 if(S.rout_hit[i0]&&S.rout_hit[i1]){
  *phit=S.phi_rout[i0]*(1.0f-f)+S.phi_rout[i1]*f;
  return 1;
 }
 int n=(f<0.5f)?i0:i1;
 if(!S.rout_hit[n])return 0;
 *phit=S.phi_rout[n];
 return 1;
}

// First intersection with the annulus rin <= r <= rout while the ray is
// inside one angular slab [ph0,ph1] of the disk wedge.  Entering the slab
// inside the annulus is a hit on the upper/lower face at that radius;
// entering the annulus radially while inside the slab is a hit on the
// outer or inner rim.
__device__ int annulus_first_hit(const Scene& S,float mu,float ph0,float ph1,float* rr_hit){
 if(ph1<ph0)return 0;
 const float uout=1.0f/S.rout,uin=1.0f/S.rin;
 float ua=path_u(S,mu,ph0);
 if(ua>=uout&&ua<=uin){*rr_hit=1.0f/ua;return 1;}
 if(ph1<=ph0)return 0;

 int nseg=min(32,max(4,(int)((ph1-ph0)*8.0f)+1));
 for(int seg=1;seg<=nseg;++seg){
  float ub=path_u(S,mu,ph0+(ph1-ph0)*(float)seg/(float)nseg);
  if(ub>0.0f){
   if(ua>0.0f){
    if(ua<uout&&ub>=uout){*rr_hit=S.rout;return 1;}
    if(ua>uin&&ub<=uin){*rr_hit=S.rin;return 1;}
   }else if(ub>=uout&&ub<=uin){*rr_hit=1.0f/ub;return 1;}
  }
  ua=ub;
 }
 return 0;
}

// Linear-sRGB radiance of a blackbody at T, normalised so that a blackbody
// at the disk's peak temperature has unit luminance.
__device__ void blackbody(const Scene& S,float T,float* c){
 float x=(logf(fmaxf(T,1.0f))-S.bb_lnT0)*S.bb_inv_dlnT;
 if(x<0.0f){c[0]=c[1]=c[2]=0.0f;return;}
 x=fminf(x,(float)(S.nbb-1)-1.0e-3f);
 int i=(int)x;
 float f=x-i;
 const float* p=S.bb_lut+4*i;
 float Y=expf(fminf(60.0f,p[3]*(1.0f-f)+p[7]*f-S.bb_lnY_ref));
 c[0]=(p[0]*(1.0f-f)+p[4]*f)*Y;
 c[1]=(p[1]*(1.0f-f)+p[5]*f)*Y;
 c[2]=(p[2]*(1.0f-f)+p[6]*f)*Y;
}

__device__ inline void set_srgb8(float* c,float r,float g,float b){
 c[0]=powf(r/255.0f,2.2f); c[1]=powf(g/255.0f,2.2f); c[2]=powf(b/255.0f,2.2f);
}

// Display-linear colour of a disk surface point.
__device__ void shade_disk(const Scene& S,float rr,int order,int rim,float lambda,float b,float* c){
 if(S.order_debug){
  if(order==0)set_srgb8(c,45,255,80);
  else if(order==1)set_srgb8(c,255,55,220);
  else if(order==2)set_srgb8(c,45,190,255);
  else if(order==3)set_srgb8(c,255,185,35);
  else set_srgb8(c,235,235,235);
  return;
 }

 // Novikov-Thorne flux, normalised to its peak (table built on the CPU).
 float t=clampf((rr-S.rin)/(S.rout-S.rin),0.0f,1.0f)*(S.nflux-1);
 int i=min((int)t,S.nflux-2);
 float f=t-i;
 float q=S.flux_lut[i]*(1.0f-f)+S.flux_lut[i+1]*f;
 if(q<=0.0f){c[0]=c[1]=c[2]=0.0f;return;}

 // Circular-geodesic emitter, static observer:
 //   g = nu_obs/nu_emit = sqrt(1-3/r) / [sqrt(f_cam) (1 - Omega lambda)],
 //   Omega = +/- r^(-3/2).
 float omega=S.rotation_sign*powf(rr,-1.5f);
 float g_true=sqrtf(fmaxf(1.0e-8f,1.0f-3.0f/rr))/(sqrtf(S.fcam)*fmaxf(1.0e-6f,1.0f-omega*lambda));
 float g=S.rel_on?clampf(g_true,0.05f,4.0f):1.0f;

 if(S.g_debug){
  // g=1 white, blueshift towards blue, redshift towards red.
  if(g>=1.0f){float s=fminf(1.0f,(g-1.0f)/0.6f);set_srgb8(c,255.0f*(1.0f-s),255.0f*(1.0f-0.35f*s),255.0f);}
  else{float s=fminf(1.0f,(1.0f-g)/0.5f);set_srgb8(c,255.0f,255.0f*(1.0f-0.65f*s),255.0f*(1.0f-s));}
  return;
 }

 // Limb darkening needs the emission angle in the gas rest frame:
 // cos = p_n/nu_emit, where p_n is the photon momentum along the surface
 // normal (unit energy at infinity).  Upper/lower faces: p_n = |p_theta|/r
 // with p_theta^2 = b^2 - lambda^2 at the equator.  Outer rim: the radial
 // component, p_n = sqrt(1 - f b^2/r^2)/sqrt(f) with f = 1 - 2/r.
 float limb=1.0f;
 if(S.limb_on){
  float fr=1.0f-2.0f/rr;
  float pn=rim?sqrtf(fmaxf(0.0f,1.0f-fr*b*b/(rr*rr))/fr)
              :sqrtf(fmaxf(0.0f,b*b-lambda*lambda))/rr;
  float cosem=pn*(S.rel_on?g_true*sqrtf(S.fcam):sqrtf(fr));
  limb=(1.0f+LIMB_K*fminf(1.0f,cosem))/(1.0f+LIMB_K*2.0f/3.0f);
 }

 if(S.color_mode==1){
  // The observed spectrum of a blackbody is exactly a blackbody at g*T
  // (I_nu/nu^3 is invariant), so no separate g^4 factor is applied here.
  float rgb[3];
  blackbody(S,S.tmax*powf(q,0.25f)*g,rgb);
  for(int k=0;k<3;++k)c[k]=1.0f-expf(-S.exposure*limb*rgb[k]);
  return;
 }

 // Bolometric intensity transforms as g^4.
 float g2=g*g;
 float mapped=1.0f-expf(-S.exposure*limb*q*g2*g2);
 if(S.color_mode==0){c[0]=c[1]=c[2]=mapped;return;}

 // Hand-painted warm palette with a restrained Doppler tint.
 float heat=powf(q,0.35f);
 float cr=1.0f,cg=0.58f+0.28f*heat,cb=0.14f+0.08f*heat;
 if(S.rel_on){
  float s=fminf(1.0f,fabsf(log2f(g))/0.8f);
  if(g>=1.0f){cr*=1.0f-0.10f*s;cg=fminf(1.0f,cg+0.04f*s);cb=fminf(1.0f,cb+0.18f*s);}
  else{cg*=1.0f-0.10f*s;cb*=1.0f-0.25f*s;}
 }
 c[0]=powf(cr,2.2f)*mapped; c[1]=powf(cg,2.2f)*mapped; c[2]=powf(cb,2.2f)*mapped;
}

// Lensed star field (and optional coordinate grid) for a ray that escapes.
__device__ void shade_sky(const Scene& S,float mu,float b,float tx,float ty,float tz,float* c){
 c[0]=c[1]=c[2]=0.0f;
 if(b<B_CRIT)return;  // black-hole shadow

 float pos=clampf((mu+1.0f)*(S.tn-1),0.0f,(float)(S.tn-1));
 int i0=(int)floorf(pos),i1=min(i0+1,S.tn-1);
 float f=pos-i0,ph;
 if(S.esc[i0]&&S.esc[i1])ph=S.pexit[i0]*(1.0f-f)+S.pexit[i1]*f;
 else{
  int n=(f<0.5f)?i0:i1;
  if(!S.esc[n])return;
  ph=S.pexit[n];
 }

 float sp,cp;
 sincosf(ph,&sp,&cp);
 float dx=cp*S.er[0]+sp*tx;
 float dy=cp*S.er[1]+sp*ty;
 float dz=cp*S.er[2]+sp*tz;

 // Sky texture pole = disk axis (+y); longitude from +z towards +x.
 float lat=asinf(clampf(dy,-1.0f,1.0f));
 float lon=atan2f(dx,dz);
 float xf=(lon/(2.0f*PI)+0.5f)*S.sw-0.5f;
 float yf=(0.5f-lat/PI)*S.sh-0.5f;
 float x0f=floorf(xf),y0f=floorf(yf);
 int x0=((int)x0f)%S.sw; if(x0<0)x0+=S.sw;
 int x1=x0+1; if(x1>=S.sw)x1=0;
 int y0=max(0,min(S.sh-1,(int)y0f));
 int y1=max(0,min(S.sh-1,(int)y0f+1));
 float wx=xf-x0f,wy=yf-y0f;
 const unsigned char* p00=S.sky+((long long)y0*S.sw+x0)*3;
 const unsigned char* p10=S.sky+((long long)y0*S.sw+x1)*3;
 const unsigned char* p01=S.sky+((long long)y1*S.sw+x0)*3;
 const unsigned char* p11=S.sky+((long long)y1*S.sw+x1)*3;
 for(int k=0;k<3;++k){
  float top=S.srgb_lut[p00[k]]*(1.0f-wx)+S.srgb_lut[p10[k]]*wx;
  float bot=S.srgb_lut[p01[k]]*(1.0f-wx)+S.srgb_lut[p11[k]]*wx;
  c[k]=top*(1.0f-wy)+bot*wy;
 }

 if(S.grid_on){
  // 15-degree graticule drawn in source-sky angles, so lensing magnifies
  // and distorts the lines exactly like the stars.
  const float spacing=PI/12.0f,halfwidth=0.0022f;
  float dlat=fabsf(lat-roundf(lat/spacing)*spacing);
  float mer=(fabsf(lat)>1.44f)?0.5f*PI:spacing;  // thin out meridians at the poles
  float dlon=fabsf(lon-roundf(lon/mer)*mer)*cosf(lat);
  float d=fminf(dlat,dlon);
  float a=1.0f-clampf((d-0.4f*halfwidth)/(0.6f*halfwidth),0.0f,1.0f);
  c[0]=fmaxf(c[0],0.016f*a); c[1]=fmaxf(c[1],0.020f*a); c[2]=fmaxf(c[2],0.040f*a);
 }
}

// Display-linear colour seen along one camera-plane sample.
__device__ void trace(const Scene& S,float px,float py,float* c){
 float rx=S.fw[0]+px*S.rt[0]+py*S.up[0];
 float ry=S.fw[1]+px*S.rt[1]+py*S.up[1];
 float rz=S.fw[2]+px*S.rt[2]+py*S.up[2];
 float inv=rsqrtf(rx*rx+ry*ry+rz*rz);
 rx*=inv; ry*=inv; rz*=inv;

 // (rx,ry,rz) is the backward ray; mu is its cosine to the outward radial
 // direction and t the transverse unit vector spanning the orbital plane.
 float mu=clampf(rx*S.er[0]+ry*S.er[1]+rz*S.er[2],-1.0f,0.0f);
 float tx=rx-mu*S.er[0],ty=ry-mu*S.er[1],tz=rz-mu*S.er[2];
 float nn=tx*tx+ty*ty+tz*tz;
 if(nn<1.0e-12f){tx=S.rt[0];ty=S.rt[1];tz=S.rt[2];}
 else{inv=rsqrtf(nn);tx*=inv;ty*=inv;tz*=inv;}

 // Conserved quantities per unit energy of the FORWARD photon: total impact
 // parameter b and angular momentum about the disk axis.
 const float scale=S.rcam/sqrtf(S.fcam);
 float b=scale*sqrtf(fmaxf(0.0f,1.0f-mu*mu));
 float lambda=-scale*(S.er[2]*rx-S.er[0]*rz);

 if(S.disk_on){
  // Height above the disk plane along the ray: n_y(phi)=A cos(phi)+B sin(phi).
  // The opaque wedge |n_y| <= sin(half-opening) is the union of slabs of
  // half-width halfw around the plane crossings c0 + k*pi.  A ray whose
  // orbital plane lies inside the wedge gets halfw = pi/2: the slabs tile
  // every angle, so the edge-on view needs no special case.
  float A=S.er[1],B=ty;
  float amp=sqrtf(A*A+B*B);
  float sh=S.disk_sin_half;
  if(amp>1.0e-8f||sh>0.0f){
   float halfw=0.0f;
   if(sh>0.0f)halfw=(amp<=sh)?0.5f*PI:asinf(sh/amp);
   float c0=atan2f(-A,B);
   if(c0>0.5f*PI)c0-=PI; else if(c0<=-0.5f*PI)c0+=PI;

   float phit=0.0f;
   int has_rout=(halfw>0.0f)?rout_crossing(S,mu,&phit):0;
   int order=0;
   for(int k=0;k<16;++k){
    float center=c0+(float)k*PI;
    float start=center-halfw,end=center+halfw;
    if(start>=S.phimax||order>S.max_order)break;
    if(end>1.0e-6f){
     // A camera inside the wedge starts in a slab (start < 0 <= end).
     start=fmaxf(start,0.0f);
     end=fminf(end,S.phimax);
     float rr=0.0f;
     int hit=0,rim=0;
     // The camera is outside r_out, so the ray is outside it until phit:
     // crossing r_out inside a slab is a hit on the outer rim.
     if(has_rout&&phit>=start&&phit<=end){rr=S.rout;hit=1;rim=1;}
     else hit=annulus_first_hit(S,mu,start,end,&rr);
     if(hit){shade_disk(S,rr,order,rim,lambda,b,c);return;}
    }
    if(center>1.0e-6f)++order;
   }
  }
 }

 shade_sky(S,mu,b,tx,ty,tz,c);
}

extern "C" __global__
void render(
 const unsigned char* sky,int sw,int sh,const float* srgb_lut,
 const unsigned char* esc,const float* pexit,int tn,
 const float* path,const unsigned char* rout_hit,const float* phi_rout,
 int nmu,int nphi,float dphi,float phimax,
 const float* flux_lut,int nflux,
 const float* bb_lut,int nbb,float bb_lnT0,float bb_inv_dlnT,float bb_lnY_ref,
 unsigned char* out,int w,int h,int ssaa,float tanx,float tany,
 const float* cam,float rcam,float rin,float rout,float disk_sin_half,
 float rotation_sign,float tmax,float exposure,
 int flags,int color_mode,int max_order
){
 int id=blockDim.x*blockIdx.x+threadIdx.x;
 if(id>=w*h)return;

 Scene S;
 S.sky=sky;S.sw=sw;S.sh=sh;S.srgb_lut=srgb_lut;
 S.esc=esc;S.pexit=pexit;S.tn=tn;
 S.path=path;S.rout_hit=rout_hit;S.phi_rout=phi_rout;
 S.nmu=nmu;S.nphi=nphi;S.dphi=dphi;S.phimax=phimax;
 S.flux_lut=flux_lut;S.nflux=nflux;
 S.bb_lut=bb_lut;S.nbb=nbb;S.bb_lnT0=bb_lnT0;S.bb_inv_dlnT=bb_inv_dlnT;S.bb_lnY_ref=bb_lnY_ref;
 for(int k=0;k<3;++k){S.er[k]=cam[k];S.fw[k]=cam[3+k];S.rt[k]=cam[6+k];S.up[k]=cam[9+k];}
 S.rcam=rcam;S.fcam=1.0f-2.0f/rcam;S.rin=rin;S.rout=rout;S.disk_sin_half=disk_sin_half;
 S.rotation_sign=rotation_sign;S.tmax=tmax;S.exposure=exposure;
 S.disk_on=flags&1;S.grid_on=(flags>>1)&1;S.rel_on=(flags>>2)&1;S.limb_on=(flags>>3)&1;
 S.order_debug=(flags>>4)&1;S.g_debug=(flags>>5)&1;
 S.color_mode=color_mode;S.max_order=max_order;

 // Supersample on a regular grid and average in linear light before the
 // single sRGB encode.
 int x=id%w,y=id/w;
 float acc[3]={0.0f,0.0f,0.0f};
 for(int sy=0;sy<ssaa;++sy){
  for(int sx=0;sx<ssaa;++sx){
   float fx=((float)x+((float)sx+0.5f)/(float)ssaa)/(float)w;
   float fy=((float)y+((float)sy+0.5f)/(float)ssaa)/(float)h;
   float c[3];
   trace(S,(fx*2.0f-1.0f)*tanx,(1.0f-fy*2.0f)*tany,c);
   acc[0]+=c[0];acc[1]+=c[1];acc[2]+=c[2];
  }
 }
 float norm=1.0f/(float)(ssaa*ssaa);
 for(int k=0;k<3;++k)out[id*3+k]=(unsigned char)(255.0f*srgb_oetf(acc[k]*norm)+0.5f);
}
"""


@dataclasses.dataclass
class View:
    """Everything that can change between frames."""
    yaw: float = 0.0
    pitch: float = math.radians(22.0)
    fov: float = 60.0
    grid: bool = True
    disk: bool = True
    color_mode: int = 1
    thick: bool = True
    half_opening_deg: float = 0.25
    relativity: bool = True
    rotation_sign: float = 1.0
    limb_darkening: bool = True
    g_map: bool = False
    order_colors: bool = False
    max_order: int = 11
    exposure: float = 1.5
    tmax: float = 4500.0
    ssaa: int = 1


class Renderer:
    def __init__(self, a, log=print):
        import cupy as cp
        self.cp = cp
        self.a = a
        name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
        self.gpu = name.decode() if isinstance(name, bytes) else str(name)

        tables = cp.RawModule(code=TABLE_CUDA)
        self._render = cp.RawKernel(RENDER_CUDA, "render")

        u_bg = 0.0 if math.isinf(a.background_r) else 1.0 / a.background_r
        self.esc = cp.zeros(a.transfer_rays, cp.uint8)
        self.pexit = cp.zeros(a.transfer_rays, cp.float32)
        threads = 256
        t = time.perf_counter()
        tables.get_function("build_transfer")(
            ((a.transfer_rays + threads - 1) // threads,), (threads,),
            (self.esc, self.pexit, np.int32(a.transfer_rays), np.float64(a.camera_r),
             np.float64(u_bg), np.float64(a.step), np.int32(int(100.0 * math.pi / a.step))))
        cp.cuda.Stream.null.synchronize()
        log(f"Sky transfer: {time.perf_counter() - t:.2f}s")

        self.dphi = a.step * a.store_every
        self.nphi = int(a.path_phi_max / self.dphi) + 1
        self.phimax = (self.nphi - 1) * self.dphi
        self.path = cp.zeros((a.path_rays, self.nphi), cp.float32)
        self.rout_hit = cp.zeros(a.path_rays, cp.uint8)
        self.phi_rout = cp.zeros(a.path_rays, cp.float32)
        threads = 128
        t = time.perf_counter()
        tables.get_function("build_path_table")(
            ((a.path_rays + threads - 1) // threads,), (threads,),
            (self.path, self.rout_hit, self.phi_rout, np.int32(a.path_rays), np.int32(self.nphi),
             np.int32(a.store_every), np.float64(a.camera_r), np.float64(a.disk_rout), np.float64(a.step)))
        cp.cuda.Stream.null.synchronize()
        log(f"Path cache: {self.path.nbytes / 1024**2:.1f} MiB, {a.path_rays}x{self.nphi}, "
            f"built in {time.perf_counter() - t:.2f}s")

        self.flux_lut = cp.asarray(phys.build_flux_lut(a.disk_rin, a.disk_rout))
        bb, self.bb_ln_t0, bb_dln_t = phys.build_blackbody_lut()
        self.bb_lut = cp.asarray(bb)
        self.bb_inv_dln_t = 1.0 / bb_dln_t
        self._bb_ref = (None, 0.0)
        self.srgb_lut = cp.asarray(phys.srgb_decode_lut())

        t = time.perf_counter()
        self.sky = cp.asarray(phys.make_sky_texture(a.seed))
        self.sky_h, self.sky_w = self.sky.shape[:2]
        log(f"Sky texture: {self.sky_w}x{self.sky_h}, generated in {time.perf_counter() - t:.2f}s")

        self.cam = cp.zeros(12, cp.float32)
        self.set_resolution(a.render_width, a.render_height)

    def set_resolution(self, width: int, height: int) -> None:
        self.w, self.h = width, height
        self.out = self.cp.empty((height, width, 3), self.cp.uint8)

    def _bb_log_luminance_ref(self, tmax: float) -> float:
        if self._bb_ref[0] != tmax:
            self._bb_ref = (tmax, phys.blackbody_log_luminance(tmax))
        return self._bb_ref[1]

    def render(self, v: View) -> np.ndarray:
        """Render one frame and return it as an (h, w, 3) uint8 sRGB array."""
        a = self.a
        self.cam.set(np.concatenate(phys.camera_basis(v.yaw, v.pitch)).astype(np.float32))
        tan_y = math.tan(math.radians(v.fov) / 2.0)
        tan_x = tan_y * self.w / self.h
        sin_half = math.sin(math.radians(v.half_opening_deg)) if v.thick else 0.0
        flags = (int(v.disk) | int(v.grid) << 1 | int(v.relativity) << 2 | int(v.limb_darkening) << 3
                 | int(v.order_colors) << 4 | int(v.g_map) << 5)
        i32, f32 = np.int32, np.float32
        threads = 256
        self._render(
            ((self.w * self.h + threads - 1) // threads,), (threads,),
            (self.sky, i32(self.sky_w), i32(self.sky_h), self.srgb_lut,
             self.esc, self.pexit, i32(a.transfer_rays),
             self.path, self.rout_hit, self.phi_rout,
             i32(a.path_rays), i32(self.nphi), f32(self.dphi), f32(self.phimax),
             self.flux_lut, i32(self.flux_lut.size),
             self.bb_lut, i32(self.bb_lut.shape[0]), f32(self.bb_ln_t0), f32(self.bb_inv_dln_t),
             f32(self._bb_log_luminance_ref(v.tmax)),
             self.out, i32(self.w), i32(self.h), i32(v.ssaa), f32(tan_x), f32(tan_y),
             self.cam, f32(a.camera_r), f32(a.disk_rin), f32(a.disk_rout), f32(sin_half),
             f32(v.rotation_sign), f32(v.tmax), f32(v.exposure),
             i32(flags), i32(v.color_mode), i32(v.max_order)))
        return self.cp.asnumpy(self.out)


def write_png(path: str, rgb: np.ndarray) -> None:
    """Minimal 8-bit RGB PNG writer, so headless output does not need pygame."""
    height, width, _ = rgb.shape
    rows = np.empty((height, 1 + width * 3), np.uint8)
    rows[:, 0] = 0
    rows[:, 1:] = rgb.reshape(height, -1)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(rows.tobytes(), 6))
                + chunk(b"IEND", b""))


def run_validations() -> None:
    """Analytical and convergence checks that need neither CUDA nor pygame."""
    checks: list[tuple[str, bool, str]] = []

    er, fw, rt, up = phys.camera_basis(0.73, -0.31)
    frame = np.array([fw, rt, up])
    checks.append(("camera basis", np.allclose(frame @ frame.T, np.eye(3), atol=1e-12)
                   and np.allclose(fw, -er), f"max |F F^T - 1| = {np.abs(frame @ frame.T - np.eye(3)).max():.1e}"))

    captured = phys.integrate_null_geodesic(phys.B_CRIT * 0.999, step=0.002)
    escaped = phys.integrate_null_geodesic(phys.B_CRIT * 1.001, step=0.002)
    checks.append(("critical capture split", not captured["escaped"] and bool(escaped["escaped"]),
                   f"0.999 b_c escaped={captured['escaped']}, 1.001 b_c escaped={escaped['escaped']}"))
    checks.append(("RK4 first integral", float(escaped["invariant_error"]) < 1.0e-10,
                   f"max error = {float(escaped['invariant_error']):.2e}"))

    worst_phi = worst_r = 0.0
    for b in (5.3, 6.0, 10.0, 20.0):
        ray = phys.integrate_null_geodesic(b, step=0.003)
        worst_phi = max(worst_phi, abs(float(ray["phi"]) - phys.exit_angle_quadrature(b)))
        worst_r = max(worst_r, abs(float(ray["r_min"]) - phys.periapsis_radius(b)))
    checks.append(("exit angle vs quadrature", worst_phi < 1.0e-7, f"max |dphi| = {worst_phi:.2e} rad"))
    checks.append(("periapsis vs cubic root", worst_r < 1.0e-7, f"max |dr| = {worst_r:.2e} M"))

    b = 1000.0
    series = 4.0 / b + 15.0 * math.pi / (4.0 * b * b)
    checks.append(("weak-field deflection", abs(phys.deflection_angle(b) - series) < 1.0e-7,
                   f"{phys.deflection_angle(b):.9f} vs 4/b + 15pi/4b^2 = {series:.9f}"))

    r_peak, _ = phys.flux_peak()
    flux_ok = phys.disk_flux_profile(phys.ISCO_R) == 0.0 and abs(r_peak - 9.5509) < 1.0e-3
    flux_ok &= abs(phys.novikov_thorne_flux(1.0e6) * 8.0e18 * math.pi / 3.0 - 1.0) < 5.0e-3
    checks.append(("Novikov-Thorne flux", bool(flux_ok), f"peak at r = {r_peak:.4f} M"))

    gp = phys.schwarzschild_g_factor(10.0, 3.0, rotation_sign=1.0)
    gm = phys.schwarzschild_g_factor(10.0, 3.0, rotation_sign=-1.0)
    mirrored = phys.schwarzschild_g_factor(10.0, -3.0, rotation_sign=1.0)
    checks.append(("rotation reversal", gp > gm and abs(gm - mirrored) < 1e-14, f"g+ = {gp:.6f}, g- = {gm:.6f}"))

    x, y = phys.blackbody_chromaticity(6500.0)
    checks.append(("blackbody chromaticity", abs(x - 0.3135) < 2e-3 and abs(y - 0.3237) < 2e-3,
                   f"6500 K -> (x, y) = ({x:.4f}, {y:.4f})"))

    failed = []
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failed.append(name)
    if failed:
        raise SystemExit("Validation failed: " + ", ".join(failed))
    print(f"All {len(checks)} Schwarzschild validation checks passed.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Real-time Schwarzschild null-geodesic ray tracer")
    g = p.add_argument_group("output")
    g.add_argument("--render-width", type=int, default=1920)
    g.add_argument("--render-height", type=int, default=1080)
    g.add_argument("--window-width", type=int, help="default: render size, fitted to the desktop")
    g.add_argument("--window-height", type=int)
    g.add_argument("--ssaa", type=int, choices=(1, 2, 3, 4),
                   help="supersampling factor per axis (default: 1, or 3 for --snapshot)")
    g.add_argument("--fps-limit", type=int, default=144)
    g.add_argument("--mouse-sensitivity", type=float, default=0.004)

    g = p.add_argument_group("scene")
    g.add_argument("--camera-r", type=float, default=30.0, help="static camera radius in M")
    g.add_argument("--background-r", type=float, default=math.inf,
                   help="radius of the star sphere in M (default: inf, stars at infinity)")
    g.add_argument("--disk-rin", type=float, default=phys.ISCO_R, help="inner disk radius, >= 6 M")
    g.add_argument("--disk-rout", type=float, default=20.0)
    g.add_argument("--disk-half-opening-deg", type=float, default=0.25)
    g.add_argument("--tmax", type=float, default=View.tmax,
                   help="peak disk temperature in K for the blackbody mode")
    g.add_argument("--exposure", type=float, default=View.exposure)
    g.add_argument("--seed", type=int, default=7, help="star field seed")

    g = p.add_argument_group("initial view (also used by --snapshot and the benchmark)")
    g.add_argument("--fov", type=float, default=60.0, help="vertical field of view in degrees")
    g.add_argument("--start-azimuth", type=float, default=0.0)
    g.add_argument("--start-elevation", type=float, default=22.0)
    g.add_argument("--color-mode", choices=COLOR_MODES, default="blackbody")
    g.add_argument("--thin-disk", action="store_true", help="razor-thin disk instead of the wedge")
    g.add_argument("--no-disk", action="store_true")
    g.add_argument("--no-grid", action="store_true")
    g.add_argument("--no-relativity", action="store_true", help="disable Doppler and gravitational shifts")
    g.add_argument("--no-limb-darkening", action="store_true")
    g.add_argument("--retrograde", action="store_true", help="reverse the disk rotation")
    g.add_argument("--g-map", action="store_true", help="show the frequency-shift map")
    g.add_argument("--order-colors", action="store_true", help="colour disk images by order")
    g.add_argument("--max-order", type=int, default=11, help="highest disk image order to draw")

    g = p.add_argument_group("geodesic tables")
    g.add_argument("--transfer-rays", type=int, default=65536)
    g.add_argument("--path-rays", type=int, default=16384)
    g.add_argument("--path-phi-max", type=float, default=12.0 * math.pi)
    g.add_argument("--store-every", type=int, default=4)
    g.add_argument("--step", type=float, default=0.003, help="RK4 step in orbital angle (rad)")

    g = p.add_argument_group("headless modes")
    g.add_argument("--validate", action="store_true", help="run CPU analytical/convergence checks and exit")
    g.add_argument("--snapshot", metavar="PNG", help="render one frame to this PNG path and exit")
    g.add_argument("--benchmark-frames", type=int, default=0, help="render N off-screen GPU frames and exit")
    g.add_argument("--benchmark-warmup", type=int, default=3, help="warm-up frames before benchmarking")
    return p


def check_args(p: argparse.ArgumentParser, a: argparse.Namespace) -> None:
    if a.disk_rin < phys.ISCO_R:
        p.error("--disk-rin must be >= 6 M: the circular-orbit disk model ends at the ISCO")
    if a.disk_rout <= a.disk_rin:
        p.error("--disk-rout must exceed --disk-rin")
    if a.camera_r <= a.disk_rout:
        p.error("--camera-r must exceed --disk-rout: the camera has to be outside the disk")
    if a.background_r <= a.camera_r:
        p.error("--background-r must exceed --camera-r")
    if not 0.0 <= a.disk_half_opening_deg <= 5.0:
        p.error("--disk-half-opening-deg must be in [0, 5]")
    if a.render_width < 16 or a.render_height < 16:
        p.error("render size must be at least 16x16")
    if not 5.0 <= a.fov <= 120.0:
        p.error("--fov must be in [5, 120] degrees")
    if a.step <= 0.0 or a.step > 0.02 or a.store_every < 1:
        p.error("--step must be in (0, 0.02] and --store-every >= 1")
    if a.transfer_rays < 1024 or a.path_rays < 1024 or a.path_phi_max < math.pi:
        p.error("table sizes are too small (need >= 1024 rays and --path-phi-max >= pi)")
    if not 500.0 <= a.tmax <= 1.0e5 or a.exposure <= 0.0:
        p.error("--tmax must be in [500, 1e5] K and --exposure positive")
    if a.benchmark_frames < 0 or a.benchmark_warmup < 0 or a.max_order < 0:
        p.error("frame counts and --max-order must be non-negative")


def view_from_args(a: argparse.Namespace) -> View:
    return View(
        yaw=math.radians(a.start_azimuth), pitch=math.radians(a.start_elevation), fov=a.fov,
        grid=not a.no_grid, disk=not a.no_disk, color_mode=COLOR_MODES.index(a.color_mode),
        thick=not a.thin_disk, half_opening_deg=a.disk_half_opening_deg,
        relativity=not a.no_relativity, rotation_sign=-1.0 if a.retrograde else 1.0,
        limb_darkening=not a.no_limb_darkening, g_map=a.g_map, order_colors=a.order_colors,
        max_order=a.max_order, exposure=a.exposure, tmax=a.tmax, ssaa=a.ssaa or 1)


def run_headless(a: argparse.Namespace) -> None:
    renderer = Renderer(a)
    view = view_from_args(a)
    if a.snapshot:
        still = dataclasses.replace(view, ssaa=a.ssaa or 3)
        write_png(a.snapshot, renderer.render(still))
        print(f"Snapshot: {a.snapshot} ({renderer.w}x{renderer.h}, SSAA {still.ssaa}x{still.ssaa})")
    if not a.benchmark_frames:
        return
    for _ in range(a.benchmark_warmup):
        renderer.render(view)
    timings = []
    for i in range(a.benchmark_frames):
        view.yaw = math.radians(a.start_azimuth) + 0.12 * i / max(1, a.benchmark_frames - 1)
        t = time.perf_counter()
        renderer.render(view)
        timings.append((time.perf_counter() - t) * 1000.0)
    values = np.asarray(timings)
    print(f"GPU: {renderer.gpu}")
    print(f"Benchmark: {renderer.w}x{renderer.h}, SSAA {view.ssaa}x{view.ssaa}, {a.benchmark_frames} frames, "
          f"median {np.median(values):.2f} ms ({1000.0 / np.median(values):.1f} FPS), "
          f"p95 {np.percentile(values, 95):.2f} ms, min {values.min():.2f} ms")


def run_interactive(a: argparse.Namespace) -> None:
    import pygame

    renderer = Renderer(a)
    pygame.init()
    # Default window: the render size, shrunk (keeping its aspect ratio) until
    # it fits the desktop with room for the title bar and taskbar.
    desktop_w, desktop_h = pygame.display.get_desktop_sizes()[0]
    fit = min(1.0, desktop_w / a.render_width, max(240, desktop_h - 90) / a.render_height)
    win_w = a.window_width or max(320, int(a.render_width * fit))
    win_h = a.window_height or max(180, int(a.render_height * fit))
    window = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
    pygame.display.set_caption(f"Schwarzschild GR ray tracer v{__version__}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    home = view_from_args(a)
    view = dataclasses.replace(home)
    order_mode = min(range(len(ORDER_LIMITS)), key=lambda i: abs(ORDER_LIMITS[i][0] - home.max_order))
    presets = {pygame.K_F1: (640, 360), pygame.K_F2: (960, 540), pygame.K_F3: (1280, 720), pygame.K_F4: (1920, 1080)}
    drag = False
    show_hud = True
    notice, notice_until = "", 0.0
    last_state, frame, surface, render_ms = None, None, None, 0.0

    print("GPU:", renderer.gpu)
    renderer.render(view)  # compile and warm the render kernel

    running = True
    while running:
        dt = min(clock.tick(a.fps_limit) / 1000.0, 0.05)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                key = e.key
                if key == pygame.K_ESCAPE:
                    running = False
                elif key == pygame.K_r:
                    view.yaw, view.pitch, view.fov = home.yaw, home.pitch, home.fov
                    view.exposure, view.tmax = home.exposure, home.tmax
                elif key == pygame.K_g:
                    view.grid = not view.grid
                elif key == pygame.K_d:
                    view.disk = not view.disk
                elif key == pygame.K_c:
                    view.color_mode = (view.color_mode + 1) % len(COLOR_MODES)
                elif key == pygame.K_h:
                    view.thick = not view.thick
                elif key == pygame.K_z:
                    view.half_opening_deg = max(0.05, view.half_opening_deg - 0.05)
                elif key == pygame.K_x:
                    view.half_opening_deg = min(5.0, view.half_opening_deg + 0.05)
                elif key == pygame.K_j:
                    view.relativity = not view.relativity
                elif key == pygame.K_v:
                    view.rotation_sign = -view.rotation_sign
                elif key == pygame.K_l:
                    view.limb_darkening = not view.limb_darkening
                elif key == pygame.K_k:
                    view.g_map = not view.g_map
                elif key == pygame.K_o:
                    view.order_colors = not view.order_colors
                elif key == pygame.K_p:
                    order_mode = (order_mode + 1) % len(ORDER_LIMITS)
                    view.max_order = ORDER_LIMITS[order_mode][0]
                elif key == pygame.K_a:
                    view.ssaa = view.ssaa % 3 + 1
                elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    view.exposure = max(0.05, view.exposure / 1.15)
                elif key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    view.exposure = min(50.0, view.exposure * 1.15)
                elif key == pygame.K_COMMA:
                    view.tmax = max(1500.0, view.tmax - 250.0)
                elif key == pygame.K_PERIOD:
                    view.tmax = min(40000.0, view.tmax + 250.0)
                elif key == pygame.K_TAB:
                    show_hud = not show_hud
                elif key == pygame.K_F12 and frame is not None:
                    name = time.strftime("screenshot_%Y%m%d_%H%M%S.png")
                    write_png(name, frame)
                    notice, notice_until = f"saved {name}", time.perf_counter() + 2.5
                elif key in presets:
                    renderer.set_resolution(*presets[key])
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                drag = True
                pygame.event.set_grab(True)
                pygame.mouse.set_visible(False)
                pygame.mouse.get_rel()
            elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                drag = False
                pygame.event.set_grab(False)
                pygame.mouse.set_visible(True)
            elif e.type == pygame.MOUSEWHEEL:
                view.fov = float(np.clip(view.fov - e.y * 2.5, 15.0, 110.0))

        keys = pygame.key.get_pressed()
        speed = math.radians(65.0) * dt
        if keys[pygame.K_q] or keys[pygame.K_LEFT]:
            view.yaw -= speed
        if keys[pygame.K_e] or keys[pygame.K_RIGHT]:
            view.yaw += speed
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            view.pitch += speed
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            view.pitch -= speed
        if drag:
            mx, my = pygame.mouse.get_rel()
            sensitivity = a.mouse_sensitivity * view.fov / 60.0
            view.yaw -= mx * sensitivity
            view.pitch += my * sensitivity
        view.pitch = float(np.clip(view.pitch, math.radians(-85.0), math.radians(85.0)))
        view.yaw = (view.yaw + math.pi) % (2.0 * math.pi) - math.pi

        # A still scene is not re-traced; the last frame is just presented again.
        state = (dataclasses.astuple(view), renderer.w, renderer.h)
        if state != last_state:
            t = time.perf_counter()
            frame = renderer.render(view)
            render_ms = (time.perf_counter() - t) * 1000.0
            surface = pygame.image.frombuffer(frame, (renderer.w, renderer.h), "RGB")
            last_state = state
        # Present letterboxed, so resizing the window never stretches the image.
        win_w, win_h = window.get_size()
        scale = min(win_w / renderer.w, win_h / renderer.h)
        shown = (max(1, round(renderer.w * scale)), max(1, round(renderer.h * scale)))
        window.fill((0, 0, 0))
        window.blit(surface if shown == (renderer.w, renderer.h) else pygame.transform.smoothscale(surface, shown),
                    ((win_w - shown[0]) // 2, (win_h - shown[1]) // 2))

        if show_hud:
            on = lambda flag: "ON" if flag else "OFF"
            lines = [
                f"Schwarzschild GR v{__version__} | {renderer.gpu} | {clock.get_fps():5.1f} FPS | render {render_ms:5.2f} ms"
                f" | {renderer.w}x{renderer.h} SSAA {view.ssaa}x",
                f"camera {a.camera_r:.1f}M  az {math.degrees(view.yaw):+.1f}  el {math.degrees(view.pitch):+.1f}"
                f"  FOV {view.fov:.1f} | disk {on(view.disk)} [{a.disk_rin:.1f}M, {a.disk_rout:.1f}M] "
                f"{'H/R ' + format(math.tan(math.radians(view.half_opening_deg)), '.4f') if view.thick else 'razor-thin'}"
                f" | spin {'+' if view.rotation_sign > 0 else '-'}Y",
                f"color {COLOR_MODES[view.color_mode].upper()} | Tmax {view.tmax:.0f} K | exposure {view.exposure:.2f}"
                f" | shifts {on(view.relativity)} | limb darkening {on(view.limb_darkening)}"
                f" | orders {ORDER_LIMITS[order_mode][1]}"
                + (" | g-map" if view.g_map else "") + (" | order colors" if view.order_colors else ""),
                "LMB/WSQE/arrows orbit | wheel FOV | C color | -/= exposure | ,/. Tmax | J shifts | V spin | L limb"
                " | H thickness | Z/X H/R",
                "D disk | G grid | K g-map | O order colors | P order limit | A SSAA | F1-F4 resolution"
                " | F12 screenshot | Tab HUD | R reset | Esc quit",
            ]
            if time.perf_counter() < notice_until:
                lines.append(notice)
            y = 8
            for text in lines:
                window.blit(font.render(text, True, (0, 0, 0)), (9, y + 1))
                window.blit(font.render(text, True, (235, 235, 235)), (8, y))
                y += 19
        pygame.display.flip()
    pygame.quit()


def main(argv=None) -> None:
    parser = build_parser()
    a = parser.parse_args(argv)
    check_args(parser, a)
    if a.validate:
        run_validations()
    elif a.snapshot or a.benchmark_frames:
        run_headless(a)
    else:
        run_interactive(a)


if __name__ == "__main__":
    main()
