// Appended to the preserved 184D kernel solely for frozen external opponents.
extern "C" __global__ void external_acceleration(
    const double* states, const unsigned char* valid, double* previous, double* accel,
    int nac, double OX,double OY,double OZ,double OS,double OC,double LS,double LC,double dt) {
    int a=blockIdx.x*blockDim.x+threadIdx.x; if(a>=nac)return;
    double s[9],R[3][3],v[3]; kin9(states+a*101,OX,OY,OZ,OS,OC,LS,LC,s);
    ned2body(s[3],s[4],s[5],R); mvT(R,s+6,v);
    for(int i=0;i<3;i++){accel[a*3+i]=valid[a]?(v[i]-previous[a*3+i])/dt:0.0; previous[a*3+i]=v[i];}
}
extern "C" __global__ void external_observation(
    const double* states,const double* accel,const float* original,float* out,int nac,
    double OX,double OY,double OZ,double OS,double OC,double LS,double LC) {
    int a=blockIdx.x*blockDim.x+threadIdx.x; if(a>=nac)return; int b=a^1;
    double s[9],t[9],Ro[3][3],Rt[3][3],vo[3],vt[3],delta[3];
    kin9(states+a*101,OX,OY,OZ,OS,OC,LS,LC,s); kin9(states+b*101,OX,OY,OZ,OS,OC,LS,LC,t);
    ned2body(s[3],s[4],s[5],Ro); ned2body(t[3],t[4],t[5],Rt);
    mvT(Ro,s+6,vo); mvT(Rt,t+6,vt); for(int i=0;i<3;i++)delta[i]=t[i]-s[i];
    double I[3][3]={{1,0,0},{0,1,0},{0,0,1}},Fo[3][3],Ft[3][3],Fl[3][3];
    dir_frame(vo,Fo); dir_frame(vt,Ft); dir_frame(delta,Fl);
    const double (*F[6])[3]={I,Ro,Rt,Fo,Ft,Fl};
    const int order[10]={0,1,2,4,5,0,1,2,3,5};
    for(int i=0;i<164;i++)out[a*214+i]=original[a*184+i];
    for(int i=0;i<10;i++){
        double v[3]; mv(F[order[i]],accel+(i<5?a:b)*3,v);
        for(int j=0;j<3;j++)STORE(out[a*214+164+i*3+j],normz(v[j],-150.0,150.0));
    }
    for(int i=0;i<20;i++)out[a*214+194+i]=original[a*184+164+i];
}
