#!/usr/bin/env python3
from __future__ import annotations
import argparse, math, time
import numpy as np

HORIZON_R=2.0
PHOTON_SPHERE_R=3.0
ISCO_R=6.0
B_CRIT=3.0*math.sqrt(3.0)

TRANSFER_CUDA=r'''
extern "C" __global__
void build_transfer(unsigned char* escaped,float* phi_exit,const int n,
 const float r_cam,const float r_bg,const float step,const int max_steps){
 int i=blockDim.x*blockIdx.x+threadIdx.x; if(i>=n)return;
 const float BC=5.196152422706632f,RH=2.0f;
 float mu=-1.0f+(float)i/(float)(n-1);
 float tr=sqrtf(fmaxf(0.0f,1.0f-mu*mu));
 if(tr<1e-7f){escaped[i]=0;phi_exit[i]=0;return;}
 float fc=1.0f-2.0f/r_cam,b=r_cam*tr/sqrtf(fc);
 if(b<BC){escaped[i]=0;phi_exit[i]=0;return;}
 float u=1.0f/r_cam,fi=1.0f/(b*b)-u*u+2.0f*u*u*u;
 float v=sqrtf(fmaxf(fi,0.0f)),phi=0.0f,ub=1.0f/r_bg,uh=1.0f/(RH*(1.0f+1e-5f));
 for(int it=0;it<max_steps;++it){
  if(u>=uh){escaped[i]=0;phi_exit[i]=phi;return;}
  if(u<=ub&&v<0){escaped[i]=1;phi_exit[i]=phi;return;}
  float k1u=v,k1v=3*u*u-u;
  float u2=u+.5f*step*k1u,v2=v+.5f*step*k1v,k2u=v2,k2v=3*u2*u2-u2;
  float u3=u+.5f*step*k2u,v3=v+.5f*step*k2v,k3u=v3,k3v=3*u3*u3-u3;
  float u4=u+step*k3u,v4=v+step*k3v,k4u=v4,k4v=3*u4*u4-u4;
  u+=(step/6)*(k1u+2*k2u+2*k3u+k4u);
  v+=(step/6)*(k1v+2*k2v+2*k3v+k4v); phi+=step;
  if(!isfinite(u)||u>10){escaped[i]=0;phi_exit[i]=phi;return;}
 }
 escaped[i]=0;phi_exit[i]=phi;
}
'''

PATH_CUDA=r"""
extern "C" __global__
void build_path_table(
 float* tab,
 unsigned char* rout_hit,
 float* phi_rout,
 const int nmu,const int nphi,const int store_every,
 const float r_cam,const float r_bg,const float disk_rout,const float step
){
 int i=blockDim.x*blockIdx.x+threadIdx.x; if(i>=nmu)return;

 float mu=-1.0f+(float)i/(float)(nmu-1);
 float tr=sqrtf(fmaxf(0.0f,1.0f-mu*mu));
 long long row=(long long)i*nphi;

 float u=1.0f/r_cam;
 tab[row]=u;
 rout_hit[i]=0;
 phi_rout[i]=0.0f;

 const float uout=1.0f/disk_rout;

 // The exactly radial ray has no useful azimuthal parameter phi (L=0),
 // but it physically reaches every smaller radius. Record the outer-disk
 // crossing analytically at phi=0 instead of leaving an empty cache row.
 if(tr<1.0e-7f){
  if(r_cam>=disk_rout){
   rout_hit[i]=1;
   phi_rout[i]=0.0f;
  }
  return;
 }

 float fc=1.0f-2.0f/r_cam;
 float b=r_cam*tr/sqrtf(fc);
 float fi=1.0f/(b*b)-u*u+2.0f*u*u*u;
 float v=sqrtf(fmaxf(fi,0.0f));

 float ub=1.0f/r_bg;
 float uh=1.0f/(2.0f*(1.0f+1e-5f));

 int s=1;
 const int maxit=(nphi-1)*store_every;

 float prev_u=u;
 float prev_phi=0.0f;

 if(u>=uout){
  rout_hit[i]=1;
  phi_rout[i]=0.0f;
 }

 for(int it=1;it<=maxit;++it){
  if(u>=uh || (u<=ub && v<0.0f))return;

  float k1u=v;
  float k1v=3.0f*u*u-u;

  float u2=u+0.5f*step*k1u;
  float v2=v+0.5f*step*k1v;
  float k2u=v2;
  float k2v=3.0f*u2*u2-u2;

  float u3=u+0.5f*step*k2u;
  float v3=v+0.5f*step*k2v;
  float k3u=v3;
  float k3v=3.0f*u3*u3-u3;

  float u4=u+step*k3u;
  float v4=v+step*k3v;
  float k4u=v4;
  float k4v=3.0f*u4*u4-u4;

  float new_u=u+(step/6.0f)*(k1u+2.0f*k2u+2.0f*k3u+k4u);
  float new_v=v+(step/6.0f)*(k1v+2.0f*k2v+2.0f*k3v+k4v);
  float new_phi=(float)it*step;

  if(!rout_hit[i] && prev_u<uout && new_u>=uout){
   float den=new_u-prev_u;
   float q=(fabsf(den)>1.0e-12f)?((uout-prev_u)/den):0.0f;
   q=fminf(1.0f,fmaxf(0.0f,q));
   rout_hit[i]=1;
   phi_rout[i]=prev_phi+q*(new_phi-prev_phi);
  }

  u=new_u;
  v=new_v;

  if(!isfinite(u)||u>10.0f)return;

  if(it%store_every==0){
   if(s>=nphi)return;
   tab[row+s]=u;
   ++s;
  }

  prev_u=u;
  prev_phi=new_phi;
 }
}
"""

RENDER_CUDA=r"""
__device__ float path_u(const float* p,int nmu,int nphi,float dphi,float mu,float ph){
 if(ph<0)return 0.0f;
 float jp=ph/dphi;
 int j0=(int)floorf(jp);
 if(j0<0||j0>=nphi-1)return 0.0f;
 int j1=j0+1;
 float fp=jp-j0;

 float mp=(mu+1.0f)*(nmu-1);
 mp=fminf(nmu-1,fmaxf(0.0f,mp));
 int i0=(int)floorf(mp),i1=min(i0+1,nmu-1);
 float fm=mp-i0;

 long long a0=(long long)i0*nphi,a1=(long long)i1*nphi;
 float a=p[a0+j0],b=p[a0+j1],c=p[a1+j0],d=p[a1+j1];

 if(a>0.0f&&b>0.0f&&c>0.0f&&d>0.0f){
  float x=a*(1.0f-fp)+b*fp;
  float y=c*(1.0f-fp)+d*fp;
  return x*(1.0f-fm)+y*fm;
 }

 // phi is singular for the exactly radial L=0 ray.
 // Use a neighboring non-radial path only for interpolation continuity.
 if((i0==0 || i1==0) && nmu>1){
  long long q=(long long)1*nphi;
  float x=p[q+j0],y=p[q+j1];
  if(x>0.0f&&y>0.0f)return x*(1.0f-fp)+y*fp;
 }

 int ii=(fm<0.5f)?i0:i1;
 long long q=(long long)ii*nphi;
 a=p[q+j0]; b=p[q+j1];
 return (a>0.0f&&b>0.0f)?a*(1.0f-fp)+b*fp:0.0f;
}

__device__ int path_annulus_first_hit(
 const float* path,int nmu,int nphi,float dphi,float mu,
 float ph0,float ph1,float rin,float rout,float* rr_hit
){
 // Find the first continuous geodesic-space intersection with the radial
 // annulus while the ray is inside one angular disk slab.  The old renderer
 // sampled only the two slab faces and its centre, which quantized the chosen
 // radius into three visibly stacked layers.  A slab is narrow, so four
 // monotonic sub-intervals plus bisection are sufficient even near periapsis.
 if(ph1<ph0)return 0;
 const float uout=1.0f/rout;
 const float uin=1.0f/rin;
 const int NSEGS=4;
 float pa=ph0;
 float ua=path_u(path,nmu,nphi,dphi,mu,pa);

 if(ua>=uout && ua<=uin){*rr_hit=1.0f/ua;return 1;}

 #pragma unroll
 for(int seg=1;seg<=NSEGS;++seg){
  float pb=ph0+(ph1-ph0)*(float)seg/(float)NSEGS;
  float ub=path_u(path,nmu,nphi,dphi,mu,pb);
  if(ub<=0.0f){pa=pb;ua=ub;continue;}

  if(ub>=uout && ub<=uin){
   // If the segment entered the annulus, solve the first crossed radial face.
   float target=(ua>uin)?uin:uout;
   if(ua>0.0f && (ua-target)*(ub-target)<=0.0f){
    float lo=pa,hi=pb,ulo=ua;
    #pragma unroll
    for(int it=0;it<7;++it){
     float pm=0.5f*(lo+hi);
     float um=path_u(path,nmu,nphi,dphi,mu,pm);
     if((ulo-target)*(um-target)<=0.0f){hi=pm;}
     else{lo=pm;ulo=um;}
    }
    float uh=path_u(path,nmu,nphi,dphi,mu,hi);
    *rr_hit=1.0f/fmaxf(uh,1.0e-12f);
   }else{
    *rr_hit=1.0f/ub;
   }
   return 1;
  }

  // The segment can cross the whole annulus between its endpoints.
  if(ua>0.0f && ua<uout && ub>uin){
   float lo=pa,hi=pb,ulo=ua;
   #pragma unroll
   for(int it=0;it<7;++it){
    float pm=0.5f*(lo+hi);
    float um=path_u(path,nmu,nphi,dphi,mu,pm);
    if((ulo-uout)*(um-uout)<=0.0f){hi=pm;}
    else{lo=pm;ulo=um;}
   }
   *rr_hit=rout;
   return 1;
  }
  if(ua>uin && ub<uout){
   float lo=pa,hi=pb,ulo=ua;
   #pragma unroll
   for(int it=0;it<7;++it){
    float pm=0.5f*(lo+hi);
    float um=path_u(path,nmu,nphi,dphi,mu,pm);
    if((ulo-uin)*(um-uin)<=0.0f){hi=pm;}
    else{lo=pm;ulo=um;}
   }
   *rr_hit=rin;
   return 1;
  }

  pa=pb;ua=ub;
 }
 return 0;
}

__device__ float intrinsic_flux_q(float rr,float rin){
 // Thin-disk-like radial flux proxy used since v0.5:
 // F ~ r^-3 [1 - sqrt(r_in/r)], normalized to its analytical maximum.
 float xx=rr/rin;
 if(xx<=1.0f)return 0.0f;
 float F=powf(xx,-3.0f)*(1.0f-rsqrtf(xx));
 const float xm=49.0f/36.0f;
 const float Fm=powf(xm,-3.0f)*(1.0f-rsqrtf(xm));
 return fminf(1.0f,fmaxf(0.0f,F/Fm));
}

__device__ void planck_rgb(float T,float* cr,float* cg,float* cb){
 // Simple three-band Planck renderer (650, 550, 450 nm).
 // This is not a full CIE color-management pipeline; it visualizes the
 // physically shifted blackbody spectrum without hand-painting a blue/red side.
 T=fminf(50000.0f,fmaxf(700.0f,T));
 const float C2=1.4387769e7f; // nm*K

 const float lr=650.0f, lg=550.0f, lb=450.0f;
 float xr=fminf(80.0f,C2/(lr*T));
 float xg=fminf(80.0f,C2/(lg*T));
 float xb=fminf(80.0f,C2/(lb*T));

 float Br=1.0f/(powf(lr,5.0f)*(expf(xr)-1.0f));
 float Bg=1.0f/(powf(lg,5.0f)*(expf(xg)-1.0f));
 float Bb=1.0f/(powf(lb,5.0f)*(expf(xb)-1.0f));

 float mx=fmaxf(Br,fmaxf(Bg,Bb));
 if(mx<=0.0f){*cr=*cg=*cb=0.0f;return;}
 *cr=Br/mx; *cg=Bg/mx; *cb=Bb/mx;
}

__device__ void gfactor_debug_rgb(float g,float* cr,float* cg,float* cb){
 // g=1 -> green/white-ish, blueshift -> blue, redshift -> red.
 if(g>=1.0f){
  float t=fminf(1.0f,(g-1.0f)/0.6f);
  *cr=1.0f-t; *cg=1.0f-0.35f*t; *cb=1.0f;
 }else{
  float t=fminf(1.0f,(1.0f-g)/0.5f);
  *cr=1.0f; *cg=1.0f-0.65f*t; *cb=1.0f-t;
 }
}

__device__ void classic_warm_rgb(float q,float g,int rel_on,float* cr,float* cg,float* cb,float* Iboost){
 float heat=powf(fminf(1.0f,fmaxf(0.0f,q)),0.35f);
 *cr=1.0f;
 *cg=0.58f+0.28f*heat;
 *cb=0.14f+0.08f*heat;
 *Iboost=1.0f;
 if(rel_on){
  float lg=logf(fmaxf(0.05f,g))/0.69314718056f;
  float t=fminf(1.0f,fabsf(lg)/0.8f);
  if(g>=1.0f){
   *cr*=1.0f-0.10f*t;
   *cg=fminf(1.0f,*cg+0.04f*t);
   *cb=fminf(1.0f,*cb+0.18f*t);
  }else{
   *cr=fminf(1.0f,*cr+0.02f*t);
   *cg*=1.0f-0.10f*t;
   *cb*=1.0f-0.25f*t;
  }
  *Iboost=powf(g,2.6f);
 }
}

__device__ void paint_disk(
 unsigned char* out,int o,float rr,float rin,int color_mode,
 int order_debug,int order,
 int rel_on,int g_debug,
 float lambda_y,float fcam,float rotation_sign,
 float tmax,float exposure
){
 if(order_debug){
  if(order==0){out[o]=45;out[o+1]=255;out[o+2]=80;}
  else if(order==1){out[o]=255;out[o+1]=55;out[o+2]=220;}
  else if(order==2){out[o]=45;out[o+1]=190;out[o+2]=255;}
  else if(order==3){out[o]=255;out[o+1]=185;out[o+2]=35;}
  else {out[o]=235;out[o+1]=235;out[o+2]=235;}
  return;
 }

 float q=intrinsic_flux_q(rr,rin);
 if(q<=0.0f){out[o]=out[o+1]=out[o+2]=0;return;}

 // Exact Schwarzschild circular-geodesic emitter:
 // Omega = +/- r^(-3/2), u^t = 1/sqrt(1-3/r).
 // Static observer at r_cam has nu_obs = E/sqrt(f_cam).
 // Therefore:
 // g = sqrt(1-3/r) / [sqrt(f_cam) (1 - Omega*lambda)]
 float g=1.0f;
 if(rel_on){
  float omega=rotation_sign*powf(rr,-1.5f);
  float denom=sqrtf(fcam)*(1.0f-omega*lambda_y);
  if(fabsf(denom)>1.0e-6f){
   g=sqrtf(fmaxf(1.0e-8f,1.0f-3.0f/rr))/denom;
  }
  g=fminf(3.0f,fmaxf(0.05f,g));
 }

 if(g_debug){
  float cr,cg,cb;
  gfactor_debug_rgb(g,&cr,&cg,&cb);
  out[o]=(unsigned char)(255.0f*cr);
  out[o+1]=(unsigned char)(255.0f*cg);
  out[o+2]=(unsigned char)(255.0f*cb);
  return;
 }

 if(color_mode==0){
  float I=q;
  if(rel_on)I*=powf(g,2.6f);
  float mapped=1.0f-expf(-exposure*I);
  mapped=powf(fminf(1.0f,fmaxf(0.0f,mapped)),1.0f/2.2f);
  unsigned char z=(unsigned char)(255.0f*mapped);
  out[o]=out[o+1]=out[o+2]=z;
  return;
 }

 if(color_mode==1){
  float cr,cg,cb,Iboost;
  classic_warm_rgb(q,g,rel_on,&cr,&cg,&cb,&Iboost);
  float I=q*Iboost;
  float mapped=1.0f-expf(-exposure*I);
  mapped=powf(fminf(1.0f,fmaxf(0.0f,mapped)),1.0f/2.2f);
  out[o]=(unsigned char)(255.0f*fminf(1.0f,cr*mapped));
  out[o+1]=(unsigned char)(255.0f*fminf(1.0f,cg*mapped));
  out[o+2]=(unsigned char)(255.0f*fminf(1.0f,cb*mapped));
  return;
 }

 // Experimental spectral mode: red/blue shift visualized via T_obs = g*T_emit.
 float Temit=tmax*powf(q,0.25f);
 float Tobs=rel_on?(g*Temit):Temit;

 float cr,cg,cb;
 planck_rgb(Tobs,&cr,&cg,&cb);

 float I=q;
 if(rel_on)I*=powf(g,4.0f);
 float mapped=1.0f-expf(-exposure*I);
 mapped=powf(fminf(1.0f,fmaxf(0.0f,mapped)),1.0f/2.2f);

 out[o]=(unsigned char)(255.0f*fminf(1.0f,cr*mapped));
 out[o+1]=(unsigned char)(255.0f*fminf(1.0f,cg*mapped));
 out[o+2]=(unsigned char)(255.0f*fminf(1.0f,cb*mapped));
}

extern "C" __global__
void render(
 const unsigned char* sky,int sw,int sh,
 const unsigned char* esc,const float* pexit,int tn,
 const float* path,const unsigned char* rout_hit,const float* phi_rout,
 int nmu,int nphi,float dphi,float phimax,
 unsigned char* out,int w,int h,float tanx,float tany,
 float erx,float ery,float erz,float fwx,float fwy,float fwz,
 float rtx,float rty,float rtz,float upx,float upy,float upz,
 float rcam,int disk_on,float rin,float rout,int color_mode,
 float disk_sin_half,int order_debug,int max_order,
 int rel_on,float rotation_sign,int g_debug,float tmax,float exposure
){
 int id=blockDim.x*blockIdx.x+threadIdx.x,n=w*h;
 if(id>=n)return;

 int x=id%w,y=id/w,o=id*3;
 float px=((((float)x+.5f)/w)*2.0f-1.0f)*tanx;
 float py=(1.0f-(((float)y+.5f)/h)*2.0f)*tany;

 float rx=fwx+px*rtx+py*upx;
 float ry=fwy+px*rty+py*upy;
 float rz=fwz+px*rtz+py*upz;
 float inv=rsqrtf(rx*rx+ry*ry+rz*rz);
 rx*=inv; ry*=inv; rz*=inv;

 float mu=fminf(0.0f,fmaxf(-1.0f,rx*erx+ry*ery+rz*erz));
 float tx=rx-mu*erx,ty=ry-mu*ery,tz=rz-mu*erz;
 float nn=tx*tx+ty*ty+tz*tz;
 if(nn<1.0e-12f){
  tx=rtx;ty=rty;tz=rtz;
 }else{
  inv=rsqrtf(nn);tx*=inv;ty*=inv;tz*=inv;
 }

 const float PI=3.14159265358979323846f;
 const float BC=5.196152422706632f;
 const float fcam=1.0f-2.0f/rcam;

 // Conserved axial impact parameter lambda = L_y/E for the FORWARD photon.
 // rx,ry,rz is the backward ray direction, hence the minus sign.
 float cross_y=erz*rx-erx*rz;
 float lambda_y=-(rcam/sqrtf(fcam))*cross_y;

 if(disk_on){
  // The disk is an optically thick equatorial wedge in world/geodesic space.
  // This is a surface/first-hit approximation, not volumetric transfer.
  float A=ery,B=ty;
  float amp=sqrtf(A*A+B*B);

  // At an exactly edge-on view some ray orbital planes lie completely inside
  // a finite wedge.  Their first physical hit is the outer radial face.  This
  // resolves the old centre hole without any screen-space mask.  The razor-
  // thin coplanar case remains deliberately undefined and falls through.
  if(disk_sin_half>0.0f && amp<=disk_sin_half){
   float mp=(mu+1.0f)*(nmu-1);
   mp=fminf(nmu-1,fmaxf(0.0f,mp));
   int mi0=(int)floorf(mp),mi1=min(mi0+1,nmu-1);
   float mf=mp-mi0;
   int nearest=(mf<0.5f)?mi0:mi1;
   int has_hit=(rout_hit[mi0]&&rout_hit[mi1])?1:rout_hit[nearest];
   if(has_hit){
    // Keep phi_rout in the calculation so the cached world-space crossing is
    // validated rather than turning this into a view-dependent special case.
    float phit=(rout_hit[mi0]&&rout_hit[mi1])?
     (phi_rout[mi0]*(1.0f-mf)+phi_rout[mi1]*mf):phi_rout[nearest];
    if(phit>=0.0f && phit<phimax){
     paint_disk(out,o,rout*(1.0f-1.0e-5f),rin,color_mode,order_debug,0,
      rel_on,g_debug,lambda_y,fcam,rotation_sign,tmax,exposure);
     return;
    }
   }
  }else if(amp>1.0e-8f){
   float halfw=0.0f;
   if(disk_sin_half>0.0f && amp>disk_sin_half){
    halfw=asinf(fminf(1.0f,disk_sin_half/amp));
   }

   float base=atan2f(-A,B);
   while(base<=1.0e-5f)base+=PI;

   #pragma unroll
   for(int k=0;k<12;++k){
    if(k>max_order)break;

    float center=base+(float)k*PI;
    float start=center-halfw;
    float end=center+halfw;
    if(end<=1.0e-5f)continue;
    if(start>=phimax)break;
    start=fmaxf(start,1.0e-5f);
    end=fminf(end,phimax);

    float rr=0.0f;
    if(path_annulus_first_hit(path,nmu,nphi,dphi,mu,start,end,rin,rout,&rr)){
     paint_disk(
      out,o,rr,rin,color_mode,order_debug,k,
      rel_on,g_debug,lambda_y,fcam,rotation_sign,tmax,exposure
     );
     return;
    }
   }
  }
 }

 // Background / black-hole shadow.
 float pos=(mu+1.0f)*(tn-1);
 pos=fminf(tn-1,fmaxf(0.0f,pos));
 int i0=(int)floorf(pos),i1=min(i0+1,tn-1);
 float f=pos-i0;
 unsigned char ok=(esc[i0]&&esc[i1])?1:esc[(f<.5f)?i0:i1];
 float ph=pexit[i0]*(1.0f-f)+pexit[i1]*f;

 float tr=sqrtf(fmaxf(0.0f,1.0f-mu*mu));
 float b=rcam*tr/sqrtf(fcam);
 if(b<BC)ok=0;

 if(!ok){
  out[o]=out[o+1]=out[o+2]=0;
  return;
 }

 float sp,cp;
 sincosf(ph,&sp,&cp);
 float dx=cp*erx+sp*tx;
 float dy=cp*ery+sp*ty;
 float dz=fminf(1.0f,fmaxf(-1.0f,cp*erz+sp*tz));

 float lon=atan2f(dy,dx);
 float lat=asinf(dz);
 float xf=(lon/(2.0f*PI)+.5f)*sw-.5f;
 float yf=(.5f-lat/PI)*sh-.5f;

 float x0f=floorf(xf),y0f=floorf(yf);
 int x0=((int)x0f)%sw;if(x0<0)x0+=sw;
 int x1=x0+1;if(x1>=sw)x1=0;
 int y0=max(0,min(sh-1,(int)y0f));
 int y1=min(sh-1,y0+1);
 float wx=xf-x0f,wy=yf-y0f;

 int p00=(y0*sw+x0)*3,p10=(y0*sw+x1)*3,p01=(y1*sw+x0)*3,p11=(y1*sw+x1)*3;
 for(int c=0;c<3;++c){
  float a=sky[p00+c]*(1.0f-wx)+sky[p10+c]*wx;
  float bb=sky[p01+c]*(1.0f-wx)+sky[p11+c]*wx;
  out[o+c]=(unsigned char)fminf(255.0f,fmaxf(0.0f,a*(1.0f-wy)+bb*wy)+.5f);
 }
}
"""

def norm(v): return v/np.linalg.norm(v)
def basis(yaw,pitch):
    cp,sp=math.cos(pitch),math.sin(pitch);cy,sy=math.cos(yaw),math.sin(yaw)
    er=np.array([sy*cp,sp,cy*cp],float);fw=-er;wu=np.array([0.,1.,0.])
    rt=np.cross(fw,wu)
    if np.linalg.norm(rt)<1e-10:rt=np.cross(fw,np.array([0.,0.,1.]))
    rt=norm(rt);up=norm(np.cross(rt,fw));return er,fw,rt,up

def disk_flux_proxy(radius: float, inner_radius: float = ISCO_R) -> float:
    """Normalized r^-3 (1-sqrt(r_in/r)) visualization emissivity."""
    x = radius / inner_radius
    if x <= 1.0:
        return 0.0
    peak_x = 49.0 / 36.0
    flux = x**-3.0 * (1.0 - x**-0.5)
    peak = peak_x**-3.0 * (1.0 - peak_x**-0.5)
    return float(np.clip(flux / peak, 0.0, 1.0))

def schwarzschild_g_factor(
    radius: float,
    lambda_axis: float,
    camera_radius: float = 30.0,
    rotation_sign: float = 1.0,
) -> float:
    """Frequency shift for a circular equatorial emitter and static observer."""
    omega = math.copysign(radius**-1.5, rotation_sign)
    numerator = math.sqrt(1.0 - 3.0 / radius)
    denominator = math.sqrt(1.0 - 2.0 / camera_radius) * (1.0 - omega * lambda_axis)
    return numerator / denominator

def integrate_null_geodesic_cpu(
    impact_parameter: float,
    camera_radius: float = 30.0,
    background_radius: float = 100.0,
    step: float = 0.001,
    max_phi: float = 40.0 * math.pi,
) -> dict[str, float | bool]:
    """Reference RK4 integrator for validation, not the real-time render path."""
    if impact_parameter <= 0.0 or camera_radius <= HORIZON_R or background_radius <= camera_radius:
        raise ValueError("invalid Schwarzschild geodesic parameters")
    u = 1.0 / camera_radius
    invariant = 1.0 / impact_parameter**2
    radial_term = invariant - u * u + 2.0 * u**3
    if radial_term < 0.0:
        raise ValueError("impact parameter does not define an inward local ray at the camera")
    v = math.sqrt(radial_term)
    phi = 0.0
    u_background = 1.0 / background_radius
    u_horizon = 1.0 / (HORIZON_R * (1.0 + 1.0e-7))
    max_u = u
    max_invariant_error = 0.0
    steps = int(max_phi / step)

    for _ in range(steps):
        if u >= u_horizon:
            return {"escaped": False, "phi": phi, "r_min": 1.0 / max_u,
                    "invariant_error": max_invariant_error}
        if u <= u_background and v < 0.0:
            return {"escaped": True, "phi": phi, "r_min": 1.0 / max_u,
                    "invariant_error": max_invariant_error}

        k1u, k1v = v, 3.0 * u * u - u
        u2, v2 = u + 0.5 * step * k1u, v + 0.5 * step * k1v
        k2u, k2v = v2, 3.0 * u2 * u2 - u2
        u3, v3 = u + 0.5 * step * k2u, v + 0.5 * step * k2v
        k3u, k3v = v3, 3.0 * u3 * u3 - u3
        u4, v4 = u + step * k3u, v + step * k3v
        k4u, k4v = v4, 3.0 * u4 * u4 - u4
        u += (step / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        v += (step / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        phi += step
        max_u = max(max_u, u)
        current = v * v + u * u - 2.0 * u**3
        max_invariant_error = max(max_invariant_error, abs(current - invariant))

    return {"escaped": False, "phi": phi, "r_min": 1.0 / max_u,
            "invariant_error": max_invariant_error}

def run_validations() -> None:
    """Fast analytical and convergence checks usable without CUDA or pygame."""
    checks: list[tuple[str, bool, str]] = []

    er, fw, rt, up = basis(0.73, -0.31)
    gram = np.array([[np.dot(a, b) for b in (er, fw, rt, up)] for a in (er, fw, rt, up)])
    checks.append(("camera basis", np.allclose(np.diag(gram), 1.0, atol=1e-12)
                   and abs(np.dot(rt, up)) < 1e-12 and np.allclose(fw, -er), str(gram)))

    peak_radius = ISCO_R * 49.0 / 36.0
    peak_ok = abs(disk_flux_proxy(peak_radius) - 1.0) < 1e-12
    peak_ok &= disk_flux_proxy(ISCO_R) == 0.0
    peak_ok &= disk_flux_proxy(20.0) < 1.0
    checks.append(("emissivity peak", peak_ok, f"r_peak={peak_radius:.9f}"))

    captured = integrate_null_geodesic_cpu(B_CRIT * 0.995, step=0.002)
    escaped = integrate_null_geodesic_cpu(B_CRIT * 1.005, step=0.002)
    checks.append(("critical capture split", not bool(captured["escaped"]) and bool(escaped["escaped"]),
                   f"below={captured['escaped']} above={escaped['escaped']}"))
    checks.append(("RK4 first integral", float(escaped["invariant_error"]) < 2.0e-8,
                   f"max error={float(escaped['invariant_error']):.3e}"))

    coarse = integrate_null_geodesic_cpu(6.0, step=0.002)
    fine = integrate_null_geodesic_cpu(6.0, step=0.001)
    convergence_error = abs(float(coarse["phi"]) - float(fine["phi"]))
    checks.append(("step convergence", bool(coarse["escaped"]) and bool(fine["escaped"])
                   and convergence_error < 0.004, f"dphi={convergence_error:.3e}"))

    gp = schwarzschild_g_factor(10.0, 3.0, rotation_sign=1.0)
    gm = schwarzschild_g_factor(10.0, 3.0, rotation_sign=-1.0)
    mirrored = schwarzschild_g_factor(10.0, -3.0, rotation_sign=1.0)
    checks.append(("rotation reversal", gp > gm and abs(gm - mirrored) < 1e-14,
                   f"g+={gp:.6f} g-={gm:.6f}"))

    failed = []
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failed.append(name)
    if failed:
        raise SystemExit("Validation failed: " + ", ".join(failed))
    print(f"All {len(checks)} Schwarzschild validation checks passed.")

def skytex(seed=7,grid=True,w=4096,h=2048):
    rng=np.random.default_rng(seed);s=np.zeros((h,w,3),np.uint8);s[:]=[1,2,6]
    yy=np.linspace(-1,1,h)[:,None];s[:,:,2]=np.clip(s[:,:,2]+(9*np.exp(-((yy-.12)/.25)**2)).astype(np.uint8),0,255)
    if grid:
        for x in range(0,w,w//24):s[:,max(0,x-1):min(w,x+2)]=np.maximum(s[:,max(0,x-1):min(w,x+2)],[12,14,22])
        for y in range(0,h,h//12):s[max(0,y-1):min(h,y+2),:]=np.maximum(s[max(0,y-1):min(h,y+2),:],[12,14,22])
    for x,y,m in zip(rng.integers(0,w,19000),rng.integers(0,h,19000),rng.random(19000)):
        val=int(55+115*m/.975) if m<.975 else int(185+70*(m-.975)/.025)
        for dx,dy,q in ((0,0,1),(-1,0,.28),(1,0,.28),(0,-1,.28),(0,1,.28),(-1,-1,.12),(1,-1,.12),(-1,1,.12),(1,1,.12)):
            xx=(int(x)+dx)%w;yy=int(y)+dy
            if 0<=yy<h:
                z=int(val*q);s[yy,xx]=np.maximum(s[yy,xx],[z,z,min(255,z+max(0,z//18))])
    return s

class Renderer:
    def __init__(self,a):
        import cupy as cp
        self.cp=cp;self.a=a
        props=cp.cuda.runtime.getDeviceProperties(0);n=props['name'];self.gpu=n.decode() if isinstance(n,bytes) else str(n)
        self.tk=cp.RawKernel(TRANSFER_CUDA,'build_transfer');self.pk=cp.RawKernel(PATH_CUDA,'build_path_table');self.rk=cp.RawKernel(RENDER_CUDA,'render')
        self.esc=cp.zeros(a.transfer_rays,cp.uint8);self.pe=cp.zeros(a.transfer_rays,cp.float32)
        self.dphi=a.step*a.store_every;self.nphi=int(a.path_phi_max/self.dphi)+1;self.phimax=(self.nphi-1)*self.dphi
        self.path=cp.zeros((a.path_rays,self.nphi),cp.float32);self.rout_hit=cp.zeros(a.path_rays,cp.uint8);self.phi_rout=cp.zeros(a.path_rays,cp.float32)
        th=256;bl=(a.transfer_rays+th-1)//th;t=time.perf_counter()
        self.tk((bl,),(th,),(self.esc,self.pe,np.int32(a.transfer_rays),np.float32(a.camera_r),np.float32(a.background_r),np.float32(a.step),np.int32(int(100*math.pi/a.step))))
        cp.cuda.Stream.null.synchronize();print(f'Sky transfer: {time.perf_counter()-t:.2f}s')
        print(f'Path cache: {self.path.nbytes/1024**2:.1f} MiB, {a.path_rays}x{self.nphi}')
        th=128;bl=(a.path_rays+th-1)//th;t=time.perf_counter()
        self.pk((bl,),(th,),(self.path,self.rout_hit,self.phi_rout,np.int32(a.path_rays),np.int32(self.nphi),np.int32(a.store_every),np.float32(a.camera_r),np.float32(a.background_r),np.float32(a.disk_rout),np.float32(a.step)))
        cp.cuda.Stream.null.synchronize();print(f'Path cache build: {time.perf_counter()-t:.2f}s')
        self.sg=cp.asarray(skytex(a.seed,True));self.sc=cp.asarray(skytex(a.seed,False));self.sh,self.sw=self.sg.shape[:2];self.setres(a.render_width,a.render_height)
    def setres(self,w,h):self.w=w;self.h=h;self.out=self.cp.empty((h,w,3),self.cp.uint8)
    def frame(self,yaw,pitch,fov,grid,disk,color_mode,thickness,order_debug,max_order,rel_on,rotation_sign,g_debug):
        er,fw,rt,up=basis(yaw,pitch);ty=math.tan(math.radians(fov)/2);tx=ty*self.w/self.h;sky=self.sg if grid else self.sc
        th=256;bl=(self.w*self.h+th-1)//th;a=self.a
        disk_sin_half=math.sin(math.radians(a.disk_half_opening_deg)) if thickness else 0.0
        self.rk((bl,),(th,),(sky,np.int32(self.sw),np.int32(self.sh),self.esc,self.pe,np.int32(a.transfer_rays),
          self.path,self.rout_hit,self.phi_rout,np.int32(a.path_rays),np.int32(self.nphi),np.float32(self.dphi),np.float32(self.phimax),
          self.out,np.int32(self.w),np.int32(self.h),np.float32(tx),np.float32(ty),
          *[np.float32(x) for x in (*er,*fw,*rt,*up)],np.float32(a.camera_r),np.int32(disk),np.float32(a.disk_rin),np.float32(a.disk_rout),np.int32(color_mode),
          np.float32(disk_sin_half),np.int32(order_debug),np.int32(max_order),
          np.int32(rel_on),np.float32(rotation_sign),np.int32(g_debug),np.float32(a.tmax),np.float32(a.exposure)))
        return self.cp.asnumpy(self.out)

def main():
    p=argparse.ArgumentParser(description='Real-time Schwarzschild null-geodesic ray tracer')
    p.add_argument('--render-width',type=int,default=1920);p.add_argument('--render-height',type=int,default=1080)
    p.add_argument('--window-width',type=int,default=1920);p.add_argument('--window-height',type=int,default=1080)
    p.add_argument('--fov',type=float,default=60);p.add_argument('--camera-r',type=float,default=30);p.add_argument('--background-r',type=float,default=100)
    p.add_argument('--disk-rin',type=float,default=6);p.add_argument('--disk-rout',type=float,default=20);p.add_argument('--disk-half-opening-deg',type=float,default=.25);p.add_argument('--start-elevation',type=float,default=22);p.add_argument('--tmax',type=float,default=9000);p.add_argument('--exposure',type=float,default=4.0)
    p.add_argument('--transfer-rays',type=int,default=65536);p.add_argument('--path-rays',type=int,default=16384);p.add_argument('--path-phi-max',type=float,default=37.6991118431)
    p.add_argument('--store-every',type=int,default=4);p.add_argument('--step',type=float,default=.003);p.add_argument('--fps-limit',type=int,default=144);p.add_argument('--mouse-sensitivity',type=float,default=.004);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--validate',action='store_true',help='run CPU analytical/convergence checks and exit')
    p.add_argument('--benchmark-frames',type=int,default=0,help='render N off-screen GPU frames and exit')
    p.add_argument('--benchmark-warmup',type=int,default=3,help='warm-up frames before benchmarking')
    p.add_argument('--snapshot',help='render one frame to this PNG path and exit')
    a=p.parse_args()
    if a.camera_r<=3 or a.background_r<=a.camera_r or a.disk_rin<2 or a.disk_rout<=a.disk_rin or a.disk_half_opening_deg<0 or a.disk_half_opening_deg>5:raise SystemExit('Invalid geometry arguments.')
    if a.benchmark_frames<0 or a.benchmark_warmup<0:raise SystemExit('Benchmark frame counts must be non-negative.')
    if a.validate:
        run_validations()
        return
    if a.benchmark_frames or a.snapshot:
        r=Renderer(a)
        yaw=0.0;pitch=math.radians(a.start_elevation)
        if a.snapshot:
            frame=r.frame(yaw,pitch,a.fov,True,True,1,True,False,11,True,1.0,False)
            import pygame
            surface=pygame.surfarray.make_surface(np.transpose(frame,(1,0,2)))
            pygame.image.save(surface,a.snapshot)
            print(f'Snapshot: {a.snapshot}')
            if not a.benchmark_frames:
                return
        for _ in range(a.benchmark_warmup):
            r.frame(yaw,pitch,a.fov,True,True,1,True,False,11,True,1.0,False)
        timings=[]
        for i in range(a.benchmark_frames):
            yaw=0.12*i/max(1,a.benchmark_frames-1)
            t=time.perf_counter()
            r.frame(yaw,pitch,a.fov,True,True,1,True,False,11,True,1.0,False)
            timings.append((time.perf_counter()-t)*1000.0)
        values=np.asarray(timings)
        print(f'GPU: {r.gpu}')
        print(f'Benchmark: {r.w}x{r.h}, {a.benchmark_frames} frames, '
              f'median {np.median(values):.2f} ms ({1000.0/np.median(values):.1f} FPS), '
              f'p95 {np.percentile(values,95):.2f} ms, min {values.min():.2f} ms')
        return
    import pygame
    r=Renderer(a);pygame.init();win=pygame.display.set_mode((a.window_width,a.window_height),pygame.RESIZABLE);pygame.display.set_caption('Schwarzschild GR v0.6.2 Hybrid Thick Disk')
    clk=pygame.time.Clock();font=pygame.font.SysFont('consolas',16);yaw=0.;pitch=math.radians(a.start_elevation);fov=a.fov;drag=False;grid=True;disk=True;color_mode=1;thickness=True;order_debug=False;order_mode=2;rel_on=True;rotation_sign=1.0;g_debug=False
    presets={pygame.K_F1:(640,360),pygame.K_F2:(960,540),pygame.K_F3:(1280,720),pygame.K_F4:(1920,1080)}
    print('GPU:',r.gpu);print('Warming render kernel...');r.frame(yaw,pitch,fov,grid,disk,color_mode,thickness,order_debug,11,rel_on,rotation_sign,g_debug)
    run=True
    while run:
        dt=min(clk.tick(a.fps_limit)/1000,.05)
        for e in pygame.event.get():
            if e.type==pygame.QUIT:run=False
            elif e.type==pygame.KEYDOWN:
                if e.key==pygame.K_ESCAPE:run=False
                elif e.key==pygame.K_r:yaw=0.;pitch=math.radians(a.start_elevation);fov=a.fov
                elif e.key==pygame.K_g:grid=not grid
                elif e.key==pygame.K_d:disk=not disk
                elif e.key==pygame.K_c:color_mode=(color_mode+1)%3
                elif e.key==pygame.K_h:thickness=not thickness
                elif e.key==pygame.K_o:order_debug=not order_debug
                elif e.key==pygame.K_p:order_mode=(order_mode+1)%3
                elif e.key==pygame.K_j:rel_on=not rel_on
                elif e.key==pygame.K_v:rotation_sign*=-1.0
                elif e.key==pygame.K_k:g_debug=not g_debug
                elif e.key in (pygame.K_LEFTBRACKET, pygame.K_z, pygame.K_COMMA, pygame.K_MINUS):
                    a.disk_half_opening_deg=max(0.02,a.disk_half_opening_deg-0.05)
                elif e.key in (pygame.K_RIGHTBRACKET, pygame.K_x, pygame.K_PERIOD, pygame.K_EQUALS):
                    a.disk_half_opening_deg=min(2.0,a.disk_half_opening_deg+0.05)
                elif e.key in presets:r.setres(*presets[e.key])
            elif e.type==pygame.MOUSEBUTTONDOWN and e.button==1:drag=True;pygame.event.set_grab(True);pygame.mouse.set_visible(False);pygame.mouse.get_rel()
            elif e.type==pygame.MOUSEBUTTONUP and e.button==1:drag=False;pygame.event.set_grab(False);pygame.mouse.set_visible(True)
            elif e.type==pygame.MOUSEWHEEL:fov=float(np.clip(fov-e.y*2.5,15,110))
        k=pygame.key.get_pressed();sp=math.radians(65)*dt
        if k[pygame.K_q] or k[pygame.K_LEFT]:yaw-=sp
        if k[pygame.K_e] or k[pygame.K_RIGHT]:yaw+=sp
        if k[pygame.K_w] or k[pygame.K_UP]:pitch+=sp
        if k[pygame.K_s] or k[pygame.K_DOWN]:pitch-=sp
        if drag:
            mx,my=pygame.mouse.get_rel();yaw-=mx*a.mouse_sensitivity;pitch+=my*a.mouse_sensitivity
        pitch=float(np.clip(pitch,math.radians(-85),math.radians(85)));yaw=(yaw+math.pi)%(2*math.pi)-math.pi
        color_name=("GRAY" if color_mode==0 else ("CLASSIC" if color_mode==1 else "SPECTRAL"));max_order=(0 if order_mode==0 else (1 if order_mode==1 else 11));t=time.perf_counter();fr=r.frame(yaw,pitch,fov,grid,disk,color_mode,thickness,order_debug,max_order,rel_on,rotation_sign,g_debug);ms=(time.perf_counter()-t)*1000
        surf=pygame.surfarray.make_surface(np.transpose(fr,(1,0,2)));ww,wh=win.get_size()
        if (ww,wh)!=(r.w,r.h):surf=pygame.transform.scale(surf,(ww,wh))
        win.blit(surf,(0,0))
        lines=[f'Schwarzschild GR v0.6.2 | {r.gpu} | {clk.get_fps():5.1f} FPS | GPU+copy {ms:5.2f} ms',
         f'render {r.w}x{r.h} | camera {a.camera_r:.1f}M | FOV {fov:.1f} | disk {"ON" if disk else "OFF"} [{a.disk_rin:.1f}M,{a.disk_rout:.1f}M] | color {color_name}',
         'LMB orbit | W/S elevation | Q/E azimuth | Wheel FOV | D disk | C color-mode | H thickness | Z/X H/R | J relativity | V reverse | K g-map | O orders | P order-limit | G grid | R reset',
         'F1 640x360 | F2 960x540 | F3 1280x720 | F4 1920x1080 | color: GRAY -> CLASSIC -> SPECTRAL | no screen-space rectangle',
         f'az {math.degrees(yaw):+.1f} deg | el {math.degrees(pitch):+.1f} deg | H/R~{math.tan(math.radians(a.disk_half_opening_deg)):.4f} {"ON" if thickness else "RAZOR"} | relativity {"ON" if rel_on else "OFF"}',
         f'rotation {("+" if rotation_sign>0 else "-")}Y | g-map {"ON" if g_debug else "OFF"} | orders: {("PRIMARY" if order_mode==0 else ("PRIMARY+SECONDARY" if order_mode==1 else "ALL"))}',
         f'path cache {a.path_rays}x{r.nphi} | primary GREEN, secondary MAGENTA, tertiary CYAN']
        yy=8
        for text in lines:
            sh=font.render(text,True,(0,0,0));fg=font.render(text,True,(235,235,235));win.blit(sh,(9,yy+1));win.blit(fg,(8,yy));yy+=19
        pygame.display.flip()
    pygame.quit()
if __name__=='__main__':main()
