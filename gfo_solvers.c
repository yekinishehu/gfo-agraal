/* gfo_solvers.c — compiled C implementation of the GFO solvers.
 * Methods: aGRAAL (PF and capped), DMPC, FOPC (He-style projection-
 * contraction).  Exact flop instrumentation + wall time via
 * clock_gettime(CLOCK_MONOTONIC).
 * Build:  gcc -O2 -std=c99 -Wall -o gfo_solvers gfo_solvers.c -lm
 */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "case_data.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static long long g_flops = 0;
#define FLOP(n) g_flops += (n)

/* circular-cone projection, z=(x1,x2,s), cone ||x|| <= mu*s */
static void proj_circ(const double *z, double mu, double *f)
{
    double nx = sqrt(z[0]*z[0] + z[1]*z[1]); FLOP(4);
    if (nx <= mu*z[2]) { f[0]=z[0]; f[1]=z[1]; f[2]=z[2]; FLOP(2); return; }
    if (nx <= -z[2]/mu) { f[0]=f[1]=f[2]=0.0; FLOP(3); return; }
    double w = (mu*nx + z[2])/(1.0 + mu*mu);
    double s = mu*w/nx;
    f[0]=s*z[0]; f[1]=s*z[1]; f[2]=w; FLOP(11);
}

/* g = -wext - A P_L(-A^T y); A is 6 x 3nf row-major; z is 3nf workspace */
static void gradF(const double *A, const double *mus, int nf, const double *wext,
                  const double *y, double *z, double *g)
{
    int n3 = 3*nf, i, j;
    for (i = 0; i < n3; i++) {
        double t = 0.0;
        for (j = 0; j < 6; j++) t += A[j*n3+i]*y[j];
        z[i] = -t; FLOP(13);
    }
    for (i = 0; i < nf; i++) proj_circ(z+3*i, mus[i], z+3*i);
    for (j = 0; j < 6; j++) {
        double t = 0.0;
        for (i = 0; i < n3; i++) t += A[j*n3+i]*z[i];
        g[j] = -wext[j] - t; FLOP(2*n3+2);
    }
}
static double dot6(const double *a, const double *b)
{ double t=0; for(int i=0;i<6;i++) t+=a[i]*b[i]; FLOP(11); return t; }
static double nrm6(const double *a){ return sqrt(dot6(a,a)); }

/* aGRAAL (Malitsky 2020 Alg. 1); returns iters, final lambda via *lam_out */
static int agraal(const double *A, const double *mus, int nf, const double *wext,
                  const double *y0, const double *y1, double lam0, double phi,
                  double lam_bar, double tol, int Kmax, double *yout,
                  double *lam_out, long long *flops_used)
{
    double rho = 1.0/phi + 1.0/(phi*phi);
    double yprev[6], y[6], ybar[6], ybar_prev[6], dprev[6], d[6], z[3*32];
    int k;
    gradF(A,mus,nf,wext,y0,z,dprev);
    gradF(A,mus,nf,wext,y1,z,d);
    double dy2=0, dd2=0;
    for(int i=0;i<6;i++){ double t=y1[i]-y0[i]; dy2+=t*t; t=d[i]-dprev[i]; dd2+=t*t; }
    FLOP(22);
    double lam = (dd2==0)? lam_bar :
        fmin(fmin(rho*lam0, phi/(4*lam0)*dy2/dd2), lam_bar);
    FLOP(8);
    memcpy(yprev,y1,sizeof yprev); memcpy(ybar_prev,y1,sizeof ybar);
    memcpy(dprev,d,sizeof dprev);
    double lam_prev = lam, th_prev = phi*lam/lam0; FLOP(2);
    for (k = 1; k < Kmax; k++) {
        for(int i=0;i<6;i++) y[i] = ybar_prev[i] - lam_prev*dprev[i];
        FLOP(12);
        gradF(A,mus,nf,wext,y,z,d);
        double dy=0, dd=0;
        for(int i=0;i<6;i++){ double t=y[i]-yprev[i]; dy+=t*t; t=d[i]-dprev[i]; dd+=t*t; }
        FLOP(22);
        lam = (dd==0)? lam_bar :
            fmin(fmin(rho*lam_prev, phi*th_prev/(4*lam_prev)*dy/dd), lam_bar);
        FLOP(8);
        for(int i=0;i<6;i++) ybar[i] = ((phi-1)*y[i]+ybar_prev[i])/phi;
        FLOP(18);
        th_prev = phi*lam/lam_prev; FLOP(2);
        memcpy(yprev,y,sizeof y); memcpy(ybar_prev,ybar,sizeof ybar);
        memcpy(dprev,d,sizeof dprev); lam_prev = lam;
        if (nrm6(d) < tol) break;
    }
    if (yout) memcpy(yout, y, sizeof y);
    if (lam_out) *lam_out = lam_prev;
    if (flops_used) *flops_used = g_flops;
    return k;
}

/* DMPC: He-style PC on grad F(y) = 0 (dual multiplier form) */
static int dmpc(const double *A, const double *mus, int nf, const double *wext,
                const double *y0, double beta0, double nu, double gam,
                double tol, int Kmax, double *yout, long long *flops_used)
{
    double y[6], yb[6], e[6], eb[6], d[6], z[3*32];
    int k;
    memcpy(y,y0,sizeof y);
    gradF(A,mus,nf,wext,y,z,e);
    double beta = beta0;
    for (k = 1; k < Kmax; k++) {
        for(;;){
            for(int i=0;i<6;i++) yb[i]=y[i]-beta*e[i];
            FLOP(12);
            gradF(A,mus,nf,wext,yb,z,eb);
            double rn=0, dn=0;
            for(int i=0;i<6;i++){ double r=y[i]-yb[i]; rn+=r*r; double t=e[i]-eb[i]; dn+=t*t; }
            FLOP(22);
            if (beta*sqrt(dn) <= nu*sqrt(rn) + 1e-300) break;
            beta *= 0.5; FLOP(3);
        }
        double rd=0, dd=0;
        for(int i=0;i<6;i++){ d[i]=(y[i]-yb[i])-beta*(e[i]-eb[i]); rd+=(y[i]-yb[i])*d[i]; dd+=d[i]*d[i]; }
        FLOP(34);
        double alpha = rd/dd; FLOP(1);
        for(int i=0;i<6;i++) y[i]-=gam*alpha*d[i];
        FLOP(18);
        gradF(A,mus,nf,wext,y,z,e);
        if (nrm6(e) < tol) break;
    }
    if (yout) memcpy(yout, y, sizeof y);
    if (flops_used) *flops_used = g_flops;
    return k;
}

/* FOPC: He-style PC on KKT system T(u)=(f+A^T lam, -A f - wext), u in L x R^6 */
static void Teval(const double *A, int nf, const double *wext,
                  const double *u, double *Tu)
{
    int n3 = 3*nf, i, j;
    for (i = 0; i < n3; i++) {
        double t = 0.0;
        for (j = 0; j < 6; j++) t += A[j*n3+i]*u[n3+j];
        Tu[i] = u[i] + t; FLOP(13);
    }
    for (j = 0; j < 6; j++) {
        double t = 0.0;
        for (i = 0; i < n3; i++) t += A[j*n3+i]*u[i];
        Tu[n3+j] = -t - wext[j]; FLOP(2*n3+2);
    }
}
static void POmega(const double *mus, int nf, double *u)
{ for (int i = 0; i < nf; i++) proj_circ(u+3*i, mus[i], u+3*i); }

static int fopc(const double *A, const double *mus, int nf, const double *wext,
                const double *u0, double beta0, double nu, double gam,
                double tol, int Kmax, double *uout, long long *flops_used)
{
    int n = 3*nf+6, i, k;
    double *u = malloc(n*8), *ub = malloc(n*8), *Tu = malloc(n*8),
           *Tub = malloc(n*8), *d = malloc(n*8), *r_ = malloc(n*8);
    memcpy(u,u0,n*8);
    Teval(A,nf,wext,u,Tu);
    double beta = beta0;
    for (k = 1; k < Kmax; k++) {
        for(;;){
            for(i=0;i<n;i++) ub[i]=u[i]-beta*Tu[i];
            FLOP(2*n);
            POmega(mus,nf,ub);
            Teval(A,nf,wext,ub,Tub);
            double rn=0, dn=0;
            for(i=0;i<n;i++){ double r=u[i]-ub[i]; rn+=r*r; double t=Tu[i]-Tub[i]; dn+=t*t; }
            FLOP(4*n);
            if (beta*sqrt(dn) <= nu*sqrt(rn) + 1e-300) break;
            beta *= 0.5; FLOP(3);
        }
        double rd=0, dd=0;
        for(i=0;i<n;i++){ r_[i]=u[i]-ub[i]; d[i]=r_[i]-beta*(Tu[i]-Tub[i]); rd+=r_[i]*d[i]; dd+=d[i]*d[i]; }
        FLOP(6*n);
        double alpha = rd/dd; FLOP(1);
        for(i=0;i<n;i++) u[i]-=gam*alpha*d[i];
        FLOP(3*n);
        Teval(A,nf,wext,u,Tu);
        for(i=0;i<n;i++) ub[i]=u[i]-Tu[i];
        FLOP(2*n);
        POmega(mus,nf,ub);
        double rn=0;
        for(i=0;i<n;i++){ double r=u[i]-ub[i]; rn+=r*r; } FLOP(3*n);
        if (sqrt(rn) < tol) break;
    }
    if (uout) memcpy(uout, u, n*8);
    free(u); free(ub); free(Tu); free(Tub); free(d); free(r_);
    if (flops_used) *flops_used = g_flops;
    return k;
}

static double now_ms(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1e3 + ts.tv_nsec*1e-6;
}

static double mat_norm(const double *A, int nf)  /* ||A|| via power it. on AA^T */
{
    int n3=3*nf, i, j, l;
    double AA[36];
    for(i=0;i<6;i++)for(j=0;j<6;j++){double t=0;for(l=0;l<n3;l++)t+=A[i*n3+l]*A[j*n3+l];AA[i*6+j]=t;}
    double v[6]={1,1,1,1,1,1}, w[6], nv=0;
    for(int it=0;it<300;it++){
        for(i=0;i<6;i++){double t=0;for(j=0;j<6;j++)t+=AA[i*6+j]*v[j];w[i]=t;}
        nv=0; for(i=0;i<6;i++) nv+=w[i]*w[i]; nv=sqrt(nv);
        for(i=0;i<6;i++) v[i]=w[i]/nv;
    }
    return nv;   /* ||A||^2 = largest eigenvalue of AA^T */
}
static double kkt_norm(const double *A, int nf)  /* ||M||, M=[I A^T;-A 0] */
{
    int n3=3*nf, n=n3+6, i, j;
    double *x = malloc(n*8), *Mx = malloc(n*8), *MM = malloc(n*8);
    for(i=0;i<n;i++) x[i]=1.0/sqrt(n);
    double nx=0;
    for(int it=0;it<400;it++){
        for(i=0;i<n3;i++){double t=0;for(j=0;j<6;j++)t+=A[j*n3+i]*x[n3+j];Mx[i]=x[i]+t;}
        for(j=0;j<6;j++){double t=0;for(i=0;i<n3;i++)t+=A[j*n3+i]*x[i];Mx[n3+j]=-t;}
        for(i=0;i<n3;i++){double t=0;for(j=0;j<6;j++)t+=A[j*n3+i]*Mx[n3+j];MM[i]=Mx[i]-t;}
        for(j=0;j<6;j++){double t=0;for(i=0;i<n3;i++)t+=A[j*n3+i]*Mx[i];MM[n3+j]=t;}
        nx=0; for(i=0;i<n;i++) nx+=MM[i]*MM[i]; nx=sqrt(nx);
        for(i=0;i<n;i++) x[i]=MM[i]/nx;
    }
    free(x); free(Mx); free(MM);
    return sqrt(nx);
}

int main(void)
{
    const double phi = 1.5;
    const double *Ac[2] = {A_A, A_B}, *mus_[2] = {MUS_A, MUS_B};
    const double *wc[2] = {WEXT_A, WEXT_B}, *ud[2] = {UDIR_A, UDIR_B};
    const int nf_[2] = {NF_A, NF_B};
    const char *nm[2] = {"A", "B"};
    for (int cs = 0; cs < 2; cs++) {
        const double *A = Ac[cs], *mus = mus_[cs], *wext = wc[cs], *udir = ud[cs];
        int nf = nf_[cs], n3 = 3*nf, n = n3+6, i;
        double LF = mat_norm(A, nf), LT = kkt_norm(A, nf);
        printf("== Case %s:  L_F = %.6f, ||M|| = %.6f ==\n", nm[cs], LF, LT);

        double y0[6]={0,0,0,0,0,0}, y1[6]={0.05,0.05,0.05,0.05,0.05,0.05};
        double out6[6]; long long fl; double t0, t1; int it;
        double tols[2]={1e-4,1e-3};
        for(int ti=0; ti<2; ti++){
          double tol=tols[ti];
          printf(" cold tol=%g:\n", tol);
          { double best=1e30; long long bfl=0; int bit=0;
            for(int r=0;r<9;r++){ g_flops=0; t0=now_ms();
              it=agraal(A,mus,nf,wext,y0,y1,1.0,phi,100.0,tol,60000,out6,NULL,&fl);
              t1=now_ms(); if(t1-t0<best){best=t1-t0;bfl=fl;bit=it;} }
            printf("  aGRAAL-PF      it=%5d  best %.4f ms  flops/iter=%lld\n",bit,best,bfl/bit); }
          { double best=1e30; long long bfl=0; int bit=0;
            for(int r=0;r<9;r++){ g_flops=0; t0=now_ms();
              it=agraal(A,mus,nf,wext,y0,y1,1.0/LF,phi,1.0/LF,tol,300000,out6,NULL,&fl);
              t1=now_ms(); if(t1-t0<best){best=t1-t0;bfl=fl;bit=it;} }
            printf("  aGRAAL-capped  it=%5d  best %.4f ms  flops/iter=%lld\n",bit,best,bfl/bit); }
          { double best=1e30; long long bfl=0; int bit=0;
            for(int r=0;r<9;r++){ g_flops=0; t0=now_ms();
              it=dmpc(A,mus,nf,wext,y0,0.9/LF,0.9,1.8,tol,300000,out6,&fl);
              t1=now_ms(); if(t1-t0<best){best=t1-t0;bfl=fl;bit=it;} }
            printf("  DMPC           it=%5d  best %.4f ms  flops/iter=%lld\n",bit,best,bfl/bit); }
          { double *u0=calloc(n,8); double *uout=malloc(n*8);
            double best=1e30; long long bfl=0; int bit=0;
            for(int r=0;r<9;r++){ g_flops=0; t0=now_ms();
              it=fopc(A,mus,nf,wext,u0,0.9/LT,0.9,1.8,tol,300000,uout,&fl);
              t1=now_ms(); if(t1-t0<best){best=t1-t0;bfl=fl;bit=it;} }
            printf("  FOPC           it=%5d  best %.4f ms  flops/iter=%lld\n",bit,best,bfl/bit);
            free(u0); free(uout); }
        }
        /* warm trajectory, T=500 */
        int T=500;
        double wn=0; for(i=0;i<6;i++) wn+=wext[i]*wext[i]; wn=sqrt(wn);
        for(int ti=0; ti<2; ti++){
          double tol=tols[ti];
          double yp[6]={0,0,0,0,0,0}; double lamp=1.0/LF, lamo;
          double tsum=0; long its=0; int itmax=0;
          for(int t=0;t<T;t++){
            double wk[6]; double s=sin(2*M_PI*t/T);
            for(i=0;i<6;i++) wk[i]=wext[i]+0.15*wn*s*udir[i];
            t0=now_ms();
            it=agraal(A,mus,nf,wk,yp,yp,lamp,phi,100.0,tol,60000,out6,&lamo,NULL);
            t1=now_ms(); tsum+=t1-t0; its+=it; if(it>itmax)itmax=it;
            memcpy(yp,out6,sizeof yp); lamp=lamo;
          }
          printf(" warm tol=%g: aGRAAL-PF mean %.1f iters, max %d, mean %.4f ms/step\n",
                 tol,(double)its/T,itmax,tsum/T);
          double yd[6]={0,0,0,0,0,0};
          tsum=0; its=0; itmax=0;
          for(int t=0;t<T;t++){
            double wk[6]; double s=sin(2*M_PI*t/T);
            for(i=0;i<6;i++) wk[i]=wext[i]+0.15*wn*s*udir[i];
            t0=now_ms();
            it=dmpc(A,mus,nf,wk,yd,0.9/LF,0.9,1.8,tol,300000,out6,NULL);
            t1=now_ms(); tsum+=t1-t0; its+=it; if(it>itmax)itmax=it;
            memcpy(yd,out6,sizeof yd);
          }
          printf(" warm tol=%g: DMPC      mean %.1f iters, max %d, mean %.4f ms/step\n",
                 tol,(double)its/T,itmax,tsum/T);
          double *up=calloc(n,8); double *uout=malloc(n*8);
          tsum=0; its=0; itmax=0;
          for(int t=0;t<T;t++){
            double wk[6]; double s=sin(2*M_PI*t/T);
            for(i=0;i<6;i++) wk[i]=wext[i]+0.15*wn*s*udir[i];
            t0=now_ms();
            it=fopc(A,mus,nf,wk,up,0.9/LT,0.9,1.8,tol,300000,uout,NULL);
            t1=now_ms(); tsum+=t1-t0; its+=it; if(it>itmax)itmax=it;
            memcpy(up,uout,n*8);
          }
          printf(" warm tol=%g: FOPC      mean %.1f iters, max %d, mean %.4f ms/step\n",
                 tol,(double)its/T,itmax,tsum/T);
          free(up); free(uout);
        }
    }
    /* per-iteration cost scaling (compiled), gradF-eval and FOPC T-eval */
    {
        const double *As[NSCALE] = {ASC_0,ASC_1,ASC_2,ASC_3,ASC_4,ASC_5};
        const double *ms[NSCALE] = {MUSC_0,MUSC_1,MUSC_2,MUSC_3,MUSC_4,MUSC_5};
        const double *ws[NSCALE] = {WEXTC_0,WEXTC_1,WEXTC_2,WEXTC_3,WEXTC_4,WEXTC_5};
        const double *ys[NSCALE] = {YC_0,YC_1,YC_2,YC_3,YC_4,YC_5};
        printf("== per-iteration cost (compiled C, microseconds) ==\n");
        for(int s=0;s<NSCALE;s++){
            int nf=SCALE_NF[s], n3=3*nf, n=n3+6;
            double z[3*32], g[6];
            double *u=calloc(n,8), *Tu=malloc(n*8);
            memcpy(u, ys[s], 48);
            gradF(As[s],ms[s],nf,ws[s],ys[s],z,g);  /* warmup */
            double t0=now_ms();
            for(int r=0;r<200000;r++) gradF(As[s],ms[s],nf,ws[s],ys[s],z,g);
            double t1=now_ms();
            g_flops=0; gradF(As[s],ms[s],nf,ws[s],ys[s],z,g); long long fg=g_flops;
            Teval(As[s],nf,ws[s],u,Tu);
            double t2=now_ms();
            for(int r=0;r<200000;r++) Teval(As[s],nf,ws[s],u,Tu);
            double t3=now_ms();
            g_flops=0; Teval(As[s],nf,ws[s],u,Tu); long long ft=g_flops;
            printf(" nf=%2d: gradF %7.3f us (%lld flops)   T-eval %7.3f us (%lld flops)\n",
                   nf,(t1-t0)*1e3/200000,fg,(t3-t2)*1e3/200000,ft);
            free(u); free(Tu);
        }
    }
    return 0;
}
