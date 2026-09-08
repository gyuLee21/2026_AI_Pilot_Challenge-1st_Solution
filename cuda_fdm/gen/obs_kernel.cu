// -*- c -*-
// claude164r 관측 + my_reward 보상 융합 커널 (fp64).
// cuda_fdm/obs_reward.py(torch, CPU 참조와 비트일치 검증됨)의 정확한 포팅.
// 두 커널: advance_kernel(1 thread/env: push+advance+종료+reward), build_obs_kernel(1 thread/기체).
// FDM 커널과 동일 규약(--fmad=false). NVRTC 단일소스(include 없음).

#define D2R 0.017453292519943295
#define R2D 57.29577951308232
#define PI2 6.283185307179586
#define FT2M 0.3048
#define M2FT 3.28084
#define SEA_LEVEL_RADIUS_FT 20925646.32546
#define ROT 7.292115e-5
#define WA 6378137.0
#define WE2 0.0066943799901411
// obs 상수(my_observation)
#define MAX_SPEED 600.0
#define MAX_RANGE_M 2500.0
#define MAX_CLOSURE 1000.0
#define VSPEED_SCALE 100.0
#define PQR_SCALE 4.0
#define ACCEL_SCALE 150.0
#define AOA_SCALE 30.0
#define SIDESLIP_SCALE 15.0
#define MIN_ALT_M 304.8
#define ALT_DANGER 300.0
#define MAX_ALT_M 15000.0
#define ENERGY_ADV 5000.0
#define PURSUIT_ATA 30.0
#define PURSUIT_RANGE 3000.0
#define FUEL_BURN 8.0e-5
#define FUEL_REF 300.0
#define REL_VEL 600.0
#define GACC 9.80665
#define EPISODE_MAX 200.0
// Same inclusive task horizon as dogfight.envs.termination (FP64 time drift).
#define TIME_LIMIT_TOLERANCE_SEC 1.0e-8
// damage
#define MIN_DMG_R_FT 500.0
#define T1_MAX 3000.0
#define T2_MAX 3500.0
#define T3_MAX 4000.0
#define T1_CONE 1.0
#define T2_CONE 2.0
#define T3_CONE 3.0
#define T2_START 100.0
#define T3_START 150.0
// final-safe reward
#define GEOM_BUDGET 5.0
#define GEOM_REF_SEC 200.0
#define SHAPING_BASE 0.0001
#define ALT_LOG_SCALE_M 304.8
#define ALT_LOG_FLOOR 1.0e-4

#define CLAMP(x,a,b) ((x)<(a)?(a):((x)>(b)?(b):(x)))

__device__ __forceinline__ void mm(const double A[3][3], const double B[3][3], double C[3][3]) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double s = 0.0;
            for (int k = 0; k < 3; k++) s += A[i][k] * B[k][j];
            C[i][j] = s;
        }
}
__device__ __forceinline__ void mv(const double A[3][3], const double v[3], double o[3]) {
    for (int i = 0; i < 3; i++) o[i] = A[i][0]*v[0] + A[i][1]*v[1] + A[i][2]*v[2];
}
__device__ __forceinline__ void mvT(const double A[3][3], const double v[3], double o[3]) {
    for (int i = 0; i < 3; i++) o[i] = A[0][i]*v[0] + A[1][i]*v[1] + A[2][i]*v[2];
}
__device__ __forceinline__ double vnorm(const double v[3]) {
    return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

// roll/pitch/yaw[deg] -> R_ned_to_body = Tx@Ty@Tz
__device__ void ned2body(double roll, double pitch, double yaw, double R[3][3]) {
    double r = roll*D2R, p = pitch*D2R, y = yaw*D2R;
    double cr = cos(r), sr = sin(r), cp = cos(p), sp = sin(p), cy = cos(y), sy = sin(y);
    double Tx[3][3] = {{1,0,0},{0,cr,sr},{0,-sr,cr}};
    double Ty[3][3] = {{cp,0,-sp},{0,1,0},{sp,0,cp}};
    double Tz[3][3] = {{cy,sy,0},{-sy,cy,0},{0,0,1}};
    double TyTz[3][3];
    mm(Ty, Tz, TyTz);
    mm(Tx, TyTz, R);
}

__device__ void quatT(double q0, double q1, double q2, double q3, double T[3][3]) {
    double q0q0 = q0*q0, q1q1 = q1*q1, q2q2 = q2*q2, q3q3 = q3*q3;
    double q0q1 = q0*q1, q0q2 = q0*q2, q0q3 = q0*q3;
    double q1q2 = q1*q2, q1q3 = q1*q3, q2q3 = q2*q3;
    T[0][0] = q0q0+q1q1-q2q2-q3q3; T[0][1] = 2.0*(q1q2+q0q3); T[0][2] = 2.0*(q1q3-q0q2);
    T[1][0] = 2.0*(q1q2-q0q3); T[1][1] = q0q0-q1q1+q2q2-q3q3; T[1][2] = 2.0*(q2q3+q0q1);
    T[2][0] = 2.0*(q1q3+q0q2); T[2][1] = 2.0*(q2q3-q0q1); T[2][2] = q0q0-q1q1-q2q2+q3q3;
}

// state[101] -> 대회 9-DOF (N/E/D[m], roll/pitch/yaw[deg], body u/v/w[m/s])
__device__ void kin9(const double* st, double OX, double OY, double OZ,
                     double OSLAT, double OCLAT, double OSLON, double OCLON, double s9[9]) {
    double px = st[0], py = st[1], pz = st[2];
    double vx = st[3], vy = st[4], vz = st[5];
    double q0 = st[6], q1 = st[7], q2 = st[8], q3 = st[9];
    double epa = st[13];
    double ce = cos(epa), se = sin(epa);
    double ex = ce*px + se*py, ey = -se*px + ce*py, ez = pz;
    double radius = sqrt(ex*ex + ey*ey + ez*ez);
    double rxy = sqrt(ex*ex + ey*ey);
    double sinLat = ez/radius, cosLat = rxy/radius, sinLon = ey/rxy, cosLon = ex/rxy;
    double Tec2l[3][3] = {
        {-cosLon*sinLat, -sinLon*sinLat, cosLat},
        {-sinLon, cosLon, 0.0},
        {-cosLon*cosLat, -sinLon*cosLat, -sinLat}};
    double Tl2ec[3][3];
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Tl2ec[i][j] = Tec2l[j][i];
    double Tec2i[3][3] = {{ce,-se,0},{se,ce,0},{0,0,1}};
    double Tl2i[3][3]; mm(Tec2i, Tl2ec, Tl2i);
    double Ti2b[3][3]; quatT(q0, q1, q2, q3, Ti2b);
    double Tl2b[3][3]; mm(Ti2b, Tl2i, Tl2b);
    double d02 = CLAMP(Tl2b[0][2], -1.0, 1.0);
    double theta = asin(-d02);
    double phi = atan2(Tl2b[1][2], Tl2b[2][2]);
    double psi = atan2(Tl2b[0][1], Tl2b[0][0]);
    if (psi < 0.0) psi += PI2;
    double dv[3] = {vx - (-ROT*py), vy - (ROT*px), vz};
    double vb[3]; mv(Ti2b, dv, vb);
    double alt_asl_m = (radius - SEA_LEVEL_RADIUS_FT) * FT2M;
    // Official bridge: tangent-origin N/E, but D is direct JSBSim MSL.
    double exm = ex*FT2M, eym = ey*FT2M, ezm = ez*FT2M;
    double b = WA * sqrt(1.0 - WE2);
    double ep2 = (WA*WA - b*b) / (b*b);
    double pxy = sqrt(exm*exm + eym*eym);
    double lon = atan2(eym, exm);
    double th = atan2(ezm*WA, pxy*b);
    double s3 = sin(th)*sin(th)*sin(th), c3 = cos(th)*cos(th)*cos(th);
    double lat = atan2(ezm + ep2*b*s3, pxy - WE2*WA*c3);
    double slat = sin(lat), clat = cos(lat), slon = sin(lon), clon = cos(lon);
    double Nrad = WA / sqrt(1.0 - WE2*slat*slat);
    double hx = (Nrad + alt_asl_m)*clat*clon;
    double hy = (Nrad + alt_asl_m)*clat*slon;
    double hz = (Nrad*(1.0 - WE2) + alt_asl_m)*slat;
    double dx = hx - OX, dy = hy - OY, dz = hz - OZ;
    s9[0] = -OSLAT*OCLON*dx - OSLAT*OSLON*dy + OCLAT*dz;
    s9[1] = -OSLON*dx + OCLON*dy;
    s9[2] = -alt_asl_m;
    s9[3] = phi*R2D; s9[4] = theta*R2D; s9[5] = psi*R2D;
    s9[6] = vb[0]*FT2M; s9[7] = vb[1]*FT2M; s9[8] = vb[2]*FT2M;
}

// 3D ATA(own->tgt), 0~180 부호없음
__device__ double ata_deg(const double so[9], const double st[9]) {
    double p[3] = {st[0]-so[0], st[1]-so[1], st[2]-so[2]};
    double n = vnorm(p);
    if (n > 0.0) { p[0]/=n; p[1]/=n; p[2]/=n; }
    double R[3][3]; ned2body(so[3], so[4], so[5], R);
    double pt[3]; mv(R, p, pt);
    return acos(CLAMP(pt[0], -1.0, 1.0)) * R2D;
}

// 3D aspect(부호있음)
__device__ double aspect_deg(const double so[9], const double st[9]) {
    double R[3][3]; ned2body(st[3], st[4], st[5], R);
    double p[3] = {so[0]-st[0], so[1]-st[1], so[2]-st[2]};
    double n = vnorm(p);
    if (n > 0.0) { p[0]/=n; p[1]/=n; p[2]/=n; }
    double b[3]; mv(R, p, b);
    double pt0 = -b[0], pt1 = -b[1], pt2 = b[2];
    double sign = (pt1 > 0.0) ? 1.0 : ((pt1 < 0.0) ? -1.0 : ((pt2 >= 0.0) ? 1.0 : -1.0));
    return sign * acos(CLAMP(pt0, -1.0, 1.0)) * R2D;
}

__device__ void los_az_el(const double so[9], const double st[9], double* az, double* el) {
    double d[3] = {st[0]-so[0], st[1]-so[1], st[2]-so[2]};
    double n = vnorm(d);
    if (n > 0.0) { d[0]/=n; d[1]/=n; d[2]/=n; }
    double R[3][3]; ned2body(so[3], so[4], so[5], R);
    double db[3]; mv(R, d, db);
    *az = atan2(db[1], db[0]) * R2D;
    *el = -asin(CLAMP(db[2], -1.0, 1.0)) * R2D;
}

__device__ void dir_frame(const double x[3], double R[3][3]) {
    double nx = vnorm(x);
    if (nx < 1e-8) {
        R[0][0]=1;R[0][1]=0;R[0][2]=0; R[1][0]=0;R[1][1]=1;R[1][2]=0; R[2][0]=0;R[2][1]=0;R[2][2]=1;
        return;
    }
    double xn[3] = {x[0]/nx, x[1]/nx, x[2]/nx};
    double refs[3][3] = {{0,0,1},{1,0,0},{0,1,0}};   // down, north, east
    double z[3]; int chosen = 2;
    for (int r = 0; r < 3; r++) {
        double d = refs[r][0]*xn[0] + refs[r][1]*xn[1] + refs[r][2]*xn[2];
        double zz[3] = {refs[r][0]-d*xn[0], refs[r][1]-d*xn[1], refs[r][2]-d*xn[2]};
        double nn = vnorm(zz);
        if (r < 2 && nn < 1e-6) continue;
        z[0]=zz[0]; z[1]=zz[1]; z[2]=zz[2]; chosen = r; break;
    }
    (void)chosen;
    double nz = vnorm(z); z[0]/=nz; z[1]/=nz; z[2]/=nz;
    double y[3] = {z[1]*xn[2]-z[2]*xn[1], z[2]*xn[0]-z[0]*xn[2], z[0]*xn[1]-z[1]*xn[0]};
    double ny = vnorm(y); y[0]/=ny; y[1]/=ny; y[2]/=ny;
    z[0]=xn[1]*y[2]-xn[2]*y[1]; z[1]=xn[2]*y[0]-xn[0]*y[2]; z[2]=xn[0]*y[1]-xn[1]*y[0];
    R[0][0]=xn[0];R[0][1]=xn[1];R[0][2]=xn[2];
    R[1][0]=y[0];R[1][1]=y[1];R[1][2]=y[2];
    R[2][0]=z[0];R[2][1]=z[1];R[2][2]=z[2];
}

__device__ void bank_sincos(const double Rb2n[3][3], const double dir[3], double* s, double* c) {
    double Rf[3][3]; dir_frame(dir, Rf);
    double by[3] = {Rb2n[0][1], Rb2n[1][1], Rb2n[2][1]};
    double yv[3]; mv(Rf, by, yv);
    double mu = atan2(yv[2], yv[1]);
    *s = sin(mu); *c = cos(mu);
}

__device__ void log_so3(const double R[3][3], double dt, double out[3]) {
    double tr = R[0][0] + R[1][1] + R[2][2];
    double cos_t = CLAMP((tr - 1.0)*0.5, -1.0, 1.0);
    double theta = acos(cos_t);
    double ax[3] = {R[2][1]-R[1][2], R[0][2]-R[2][0], R[1][0]-R[0][1]};
    double denom = 2.0*sin(theta);
    double scale = 0.0;
    if (theta >= 1e-8 && fabs(denom) >= 1e-8) scale = theta / denom;
    double inv = 1.0 / (dt > 1e-8 ? dt : 1e-8);
    out[0] = ax[0]*scale*inv; out[1] = ax[1]*scale*inv; out[2] = ax[2]*scale*inv;
}

__device__ __forceinline__ double normz(double x, double lo, double hi) {
    if (hi <= lo) return 0.0;
    double mid = (hi+lo)*0.5, half = (hi-lo)*0.5;
    return (CLAMP(x, lo, hi) - mid) / half;
}

__device__ double damage(double r_ft, double ata_abs, double t) {
    double a = fabs(ata_abs);
    if (r_ft >= MIN_DMG_R_FT && r_ft <= T1_MAX && a < T1_CONE)
        return 1.0*(T1_MAX - r_ft)/(T1_MAX - MIN_DMG_R_FT);
    if (t >= T2_START && r_ft >= MIN_DMG_R_FT && r_ft <= T2_MAX && a < T2_CONE)
        return 0.3*(T2_MAX - r_ft)/(T2_MAX - MIN_DMG_R_FT);
    if (t >= T3_START && r_ft >= MIN_DMG_R_FT && r_ft <= T3_MAX && a < T3_CONE)
        return 0.1*(T3_MAX - r_ft)/(T3_MAX - MIN_DMG_R_FT);
    return 0.0;
}

__device__ __forceinline__ double smoothstep3(double x, double a, double b) {
    double z = CLAMP((x-a)/(b-a), 0.0, 1.0);
    return z*z*(3.0-2.0*z);
}

__device__ void unit_fallback(const double v[3], const double fb[3], double out[3]) {
    double n = vnorm(v);
    const double* src = v;
    if (!(n >= 1e-9) || !isfinite(n)) { src = fb; n = vnorm(fb); }
    if (!(n >= 1e-9) || !isfinite(n)) {
        out[0]=1.0; out[1]=0.0; out[2]=0.0; return;
    }
    out[0]=src[0]/n; out[1]=src[1]/n; out[2]=src[2]/n;
}

__device__ double align3(const double a[3], const double b[3]) {
    const double ex[3] = {1.0,0.0,0.0};
    double au[3], bu[3]; unit_fallback(a, ex, au); unit_fallback(b, au, bu);
    double q = 0.5*(1.0 + CLAMP(au[0]*bu[0]+au[1]*bu[1]+au[2]*bu[2], -1.0, 1.0));
    return CLAMP(q*q*q, 0.0, 1.0);
}

__device__ void velocity_ned(const double s[9], double vn[3], double forward[3]) {
    double Rnb[3][3]; ned2body(s[3], s[4], s[5], Rnb);
    double vb[3] = {s[6],s[7],s[8]}; mvT(Rnb, vb, vn);
    // body x-axis expressed in NED = first row of R_ned_to_body.
    forward[0]=Rnb[0][0]; forward[1]=Rnb[0][1]; forward[2]=Rnb[0][2];
}

__device__ double safe_lead_score(const double so[9], const double st[9],
                                  const double own_v[3], const double tgt_v[3],
                                  const double forward[3], double dist_m,
                                  double dist_ft, double ata) {
    double rel[3] = {st[0]-so[0],st[1]-so[1],st[2]-so[2]};
    double los[3]; unit_fallback(rel, forward, los);
    double path[3]; unit_fallback(own_v, forward, path);
    double speed = vnorm(own_v); if (speed < 30.0) speed = 30.0;
    double tau = CLAMP(dist_m/speed, 0.30, 1.50);
    double leadv[3] = {rel[0]+tgt_v[0]*tau, rel[1]+tgt_v[1]*tau, rel[2]+tgt_v[2]*tau};
    double lead[3]; unit_fallback(leadv, los, lead);
    double intercept = 0.40*align3(forward,lead) + 0.60*align3(path,lead);
    double direct = align3(forward,los);
    double w = smoothstep3(dist_ft,1000.0,2500.0)*smoothstep3(fabs(ata),3.0,8.0);
    return CLAMP(w*intercept+(1.0-w)*direct,0.0,1.0);
}

__device__ double broad_score(double ata) {
    ata = CLAMP(fabs(ata),0.0,180.0);
    return (ata <= 30.0) ? 1.0 : CLAMP((180.0-ata)/150.0,0.0,1.0);
}

__device__ double geometry_side(const double so[9], const double st[9],
                                const double own_v[3], const double tgt_v[3],
                                const double forward[3], double dist_m, double dist_ft,
                                double ata, double t, double range_gate,
                                double closure_quality) {
    double lead = safe_lead_score(so,st,own_v,tgt_v,forward,dist_m,dist_ft,ata);
    double angle_gate = 1.0-smoothstep3(fabs(ata),3.0,8.0);
    double control = CLAMP(angle_gate*range_gate*closure_quality,0.0,1.0);
    double fine = damage(dist_ft,fabs(ata),t);
    return CLAMP(0.50*broad_score(ata)+0.20*lead+0.20*control+0.10*fine,0.0,1.0);
}

__device__ double geometry_advantage(const double so[9], const double st[9], double t) {
    double rel[3] = {st[0]-so[0],st[1]-so[1],st[2]-so[2]};
    double dist_m = vnorm(rel), dist_ft = dist_m/FT2M;
    const double ex[3]={1.0,0.0,0.0}; double los[3]; unit_fallback(rel,ex,los);
    double vo[3],vp[3],fo[3],fp[3];
    velocity_ned(so,vo,fo); velocity_ned(st,vp,fp);
    double closing = (vo[0]-vp[0])*los[0]+(vo[1]-vp[1])*los[1]+(vo[2]-vp[2])*los[2];
    double mean_speed=0.5*(vnorm(vo)+vnorm(vp)); if(mean_speed<100.0) mean_speed=100.0;
    double ratio=closing/mean_speed;
    double desired=0.20*smoothstep3(dist_ft,2000.0,4500.0)
                  -0.12*(1.0-smoothstep3(dist_ft,650.0,900.0));
    double dq=(ratio-desired)/0.15;
    double closure_quality=exp(-0.5*dq*dq);
    double range_gate=smoothstep3(dist_ft,600.0,900.0)
                     *(1.0-smoothstep3(dist_ft,3200.0,4300.0));
    double a1=ata_deg(so,st), a2=ata_deg(st,so);
    double S1=geometry_side(so,st,vo,vp,fo,dist_m,dist_ft,a1,t,range_gate,closure_quality);
    double S2=geometry_side(st,so,vp,vo,fp,dist_m,dist_ft,a2,t,range_gate,closure_quality);
    return CLAMP(S1-S2,-1.0,1.0);
}

__device__ double altitude_log(double altitude_m) {
    double scaled=altitude_m/ALT_LOG_SCALE_M;
    return log(scaled<ALT_LOG_FLOOR ? ALT_LOG_FLOOR : scaled);
}

#define STORE(dst, val) { double _v = (val); \
    if (!isfinite(_v)) _v = (_v > 0.0) ? 10.0 : ((_v < 0.0) ? -10.0 : 0.0); \
    (dst) = (float)_v; }

// 관점 기체 obs 214 를 out 에 기록. so=own9, st=tgt9.
__device__ void build_obs_one(const double so[9], const double st[9],
                              double hp_o, double hp_t, double fuel_o, double fuel_t,
                              const double pqr_o[3], const double pqr_p[3],
                              const double accel_o[3], const double accel_p[3],
                              double dmg_dealt, double dmg_taken, double t_ac,
                              const double* acth, float* out, double* aux) {
    double Rnb_o[3][3], Rnb_t[3][3];
    ned2body(so[3], so[4], so[5], Rnb_o);
    ned2body(st[3], st[4], st[5], Rnb_t);
    double ovb[3] = {so[6], so[7], so[8]}, tvb[3] = {st[6], st[7], st[8]};
    double own_vn[3], tgt_vn[3];
    mvT(Rnb_o, ovb, own_vn);
    mvT(Rnb_t, tvb, tgt_vn);
    // FUTURE_AUX_BEGIN: reuse existing kinematics; no extra observation or kernel.
    if (aux) {
        for (int i=0;i<3;i++) {
            aux[i]=so[i]; aux[3+i]=own_vn[i];
            aux[6+i]=st[i]; aux[9+i]=tgt_vn[i];
            for (int j=0;j<3;j++) aux[12+3*i+j]=Rnb_o[i][j];
        }
    }
    // FUTURE_AUX_END
    double rel_vn[3] = {tgt_vn[0]-own_vn[0], tgt_vn[1]-own_vn[1], tgt_vn[2]-own_vn[2]};
    double own_spd = vnorm(ovb), tgt_spd = vnorm(tvb);
    double own_alt = -so[2], tgt_alt = -st[2];
    double delta[3] = {st[0]-so[0], st[1]-so[1], st[2]-so[2]};
    double dist = vnorm(delta);
    double los_u[3] = {0,0,0};
    double closure = 0.0;
    if (dist > 1e-6) {
        los_u[0]=delta[0]/dist; los_u[1]=delta[1]/dist; los_u[2]=delta[2]/dist;
        closure = (own_vn[0]-tgt_vn[0])*los_u[0] + (own_vn[1]-tgt_vn[1])*los_u[1]
                + (own_vn[2]-tgt_vn[2])*los_u[2];
    }
    double ata = ata_deg(so, st);
    double enemy_ata = ata_deg(st, so);
    double aa = aspect_deg(so, st);
    double az, el; los_az_el(so, st, &az, &el);

    double u = ovb[0], v = ovb[1], w = ovb[2];
    double aoa = 0.0, sslip = 0.0;
    if (own_spd >= 1.0) {
        aoa = atan2(w, u) * R2D;
        sslip = atan2(v, sqrt(u*u + w*w)) * R2D;
    }
    double vspeed = -own_vn[2];
    double e_own = own_alt + own_spd*own_spd/(2.0*GACC);
    double e_tgt = tgt_alt + tgt_spd*tgt_spd/(2.0*GACC);
    double e_adv = e_own - e_tgt;

    double cone = (t_ac >= T3_START) ? T3_CONE : ((t_ac >= T2_START) ? T2_CONE : T1_CONE);
    double maxrng_ft = (t_ac >= T3_START) ? T3_MAX : ((t_ac >= T2_START) ? T2_MAX : T1_MAX);
    double aim_sharp = 2.0*exp(-((ata/3.0)*(ata/3.0))) - 1.0;
    double aim_margin = tanh((cone - fabs(ata)) / (cone > 1e-6 ? cone : 1e-6));
    double en_aim_sharp = 2.0*exp(-((enemy_ata/3.0)*(enemy_ata/3.0))) - 1.0;
    double en_aim_margin = tanh((cone - fabs(enemy_ata)) / (cone > 1e-6 ? cone : 1e-6));
    double min_r_m = MIN_DMG_R_FT*FT2M, max_r_m = maxrng_ft*FT2M;
    double span = (max_r_m - min_r_m); if (span < 1e-6) span = 1e-6;
    double rm_near = tanh((dist - min_r_m)/span);
    double rm_far = tanh((max_r_m - dist)/span);

    double ovbank_s, ovbank_c, tvbank_s, tvbank_c;
    double Rb2n_o[3][3], Rb2n_t[3][3];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++){ Rb2n_o[i][j]=Rnb_o[j][i]; Rb2n_t[i][j]=Rnb_t[j][i]; }
    bank_sincos(Rb2n_o, own_vn, &ovbank_s, &ovbank_c);
    bank_sincos(Rb2n_t, tgt_vn, &tvbank_s, &tvbank_c);

    double pf_ata = 1.0 - fabs(ata)/PURSUIT_ATA; if (pf_ata < 0.0) pf_ata = 0.0;
    double pf_rng = 1.0 - dist/PURSUIT_RANGE; if (pf_rng < 0.0) pf_rng = 0.0;
    double pursuit = 2.0*(pf_ata*pf_rng) - 1.0;

    // ── 스칼라 50 ──
    STORE(out[0], normz(own_spd, 0.0, MAX_SPEED));
    STORE(out[1], normz(tgt_spd, 0.0, MAX_SPEED));
    STORE(out[2], tanh(aoa/AOA_SCALE));
    STORE(out[3], tanh(sslip/SIDESLIP_SCALE));
    STORE(out[4], tanh((own_alt - MIN_ALT_M)/ALT_DANGER));
    STORE(out[5], normz(vspeed, -VSPEED_SCALE, VSPEED_SCALE));
    STORE(out[6], normz(hp_o, 0.0, 1.0));
    STORE(out[7], normz(hp_t, 0.0, 1.0));
    STORE(out[8], hp_o - hp_t);
    STORE(out[9], e_adv/(fabs(e_adv) + ENERGY_ADV));
    STORE(out[10], normz(dist, 0.0, MAX_RANGE_M));
    STORE(out[11], normz(closure, -MAX_CLOSURE, MAX_CLOSURE));
    STORE(out[12], sin(ata*D2R)); STORE(out[13], cos(ata*D2R));
    STORE(out[14], sin(aa*D2R));  STORE(out[15], cos(aa*D2R));
    STORE(out[16], sin(az*D2R));  STORE(out[17], cos(az*D2R));
    STORE(out[18], sin(el*D2R));  STORE(out[19], cos(el*D2R));
    STORE(out[20], aim_sharp);
    STORE(out[21], aim_margin);
    STORE(out[22], en_aim_sharp);
    STORE(out[23], en_aim_margin);
    STORE(out[24], rm_near);
    STORE(out[25], rm_far);
    STORE(out[26], normz(t_ac, 0.0, EPISODE_MAX));
    STORE(out[27], sin(so[3]*D2R)); STORE(out[28], cos(so[3]*D2R));
    STORE(out[29], sin(so[4]*D2R)); STORE(out[30], cos(so[4]*D2R));
    STORE(out[31], sin(so[5]*D2R)); STORE(out[32], cos(so[5]*D2R));
    STORE(out[33], sin(st[3]*D2R)); STORE(out[34], cos(st[3]*D2R));
    STORE(out[35], sin(st[4]*D2R)); STORE(out[36], cos(st[4]*D2R));
    STORE(out[37], sin(st[5]*D2R)); STORE(out[38], cos(st[5]*D2R));
    STORE(out[39], ovbank_s); STORE(out[40], ovbank_c);
    STORE(out[41], tvbank_s); STORE(out[42], tvbank_c);
    STORE(out[43], normz(fuel_o, 0.0, 1.0));
    STORE(out[44], normz(fuel_t, 0.0, 1.0));
    STORE(out[45], CLAMP(2.0*dmg_dealt - 1.0, -1.0, 1.0));
    STORE(out[46], CLAMP(2.0*dmg_taken - 1.0, -1.0, 1.0));
    STORE(out[47], pursuit);
    STORE(out[48], normz(own_alt, 0.0, MAX_ALT_M));
    STORE(out[49], normz(tgt_alt, 0.0, MAX_ALT_M));

    // ── 벡터 144 (VEC_LAYOUT 순서: ...omega 다음에 own_accel, tgt_accel 30 추가) ──
    double own_om_n[3], tgt_om_n[3];
    mvT(Rnb_o, pqr_o, own_om_n);
    mvT(Rnb_t, pqr_p, tgt_om_n);
    double I3[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    double Fmy[3][3], Fopp[3][3], Flos[3][3];
    dir_frame(own_vn, Fmy); dir_frame(tgt_vn, Fopp); dir_frame(delta, Flos);
    // frame ptr 배열
    const double (*F[6])[3] = {I3, Rnb_o, Rnb_t, Fmy, Fopp, Flos};
    double grav[3] = {0,0,1};
    // accel_o/accel_p 는 advance_kernel 이 이미 NED 로 적분해 넘겨준다(body->NED 변환 불요).
    const double* V[9] = {grav, los_u, own_vn, tgt_vn, rel_vn, own_om_n, tgt_om_n,
                          accel_o, accel_p};
    const int LAY[48][3] = {
        {0,1,0},{0,2,0},{0,3,0},{0,4,0},{0,5,0},
        {1,0,0},{1,1,0},{1,2,0},{1,3,0},{1,4,0},
        {2,0,1},{2,1,1},{2,2,1},{2,4,1},{2,5,1},
        {3,0,1},{3,1,1},{3,2,1},{3,3,1},{3,5,1},
        {4,0,1},{4,1,1},{4,2,1},{4,3,1},{4,4,1},{4,5,1},
        {5,0,2},{5,1,2},{5,2,2},{5,3,2},{5,4,2},{5,5,2},
        {6,0,2},{6,1,2},{6,2,2},{6,3,2},{6,4,2},{6,5,2},
        {7,0,3},{7,1,3},{7,2,3},{7,4,3},{7,5,3},
        {8,0,3},{8,1,3},{8,2,3},{8,3,3},{8,5,3}};
    int base = 50;
    for (int i = 0; i < 48; i++) {
        int vi = LAY[i][0], fi = LAY[i][1], kind = LAY[i][2];
        const double (*Rm)[3] = F[fi];
        const double* vv = V[vi];
        double comp[3];
        for (int r = 0; r < 3; r++) comp[r] = Rm[r][0]*vv[0] + Rm[r][1]*vv[1] + Rm[r][2]*vv[2];
        for (int r = 0; r < 3; r++) {
            double val;
            if (kind == 0) val = comp[r];
            else if (kind == 1) val = normz(comp[r], -REL_VEL, REL_VEL);
            else if (kind == 3) val = normz(comp[r], -ACCEL_SCALE, ACCEL_SCALE);
            else val = tanh(comp[r]/PQR_SCALE);
            STORE(out[base + i*3 + r], val);
        }
    }
    // ── action history 20 ──
    for (int j = 0; j < 20; j++) STORE(out[194 + j], acth[j]);
}

// reset/autoreset 직후 최초 상태의 G를 저장해 첫 RL interval도 사다리꼴로 적분한다.
extern "C" __global__ void init_reward_kernel(
    const double* states, const unsigned char* env_mask,
    double* prev_geometry, unsigned char* prev_geometry_valid, double* prev_alt_log,
    const double* t_sec,
    int nenv,
    double OX, double OY, double OZ, double OSLAT, double OCLAT, double OSLON, double OCLON) {
    int e=blockIdx.x*blockDim.x+threadIdx.x;
    if(e>=nenv || !env_mask[e]) return;
    int o=2*e,p=o+1; double so[9],sp[9];
    kin9(states+o*101,OX,OY,OZ,OSLAT,OCLAT,OSLON,OCLON,so);
    kin9(states+p*101,OX,OY,OZ,OSLAT,OCLAT,OSLON,OCLON,sp);
    double g=geometry_advantage(so,sp,t_sec[e]);
    prev_geometry[o]=g; prev_geometry[p]=-g;
    prev_geometry_valid[o]=1; prev_geometry_valid[p]=1;
    // Seed both perspectives from this episode's initial state (no reset impulse).
    prev_alt_log[o]=altitude_log(-so[2]); prev_alt_log[p]=altitude_log(-sp[2]);
}

// ────────────────────────────────────────────────────────────────────────────
extern "C" __global__ void advance_kernel(
    const double* states, const double* actions,
    double* hp, double* fuel, double* t_sec,
    double* prev_att, unsigned char* prev_valid, double* pqr,
    double* prev_vel_ned, double* accel,
    double* last_dmg_dealt, double* last_dmg_taken, double* hp_loss,
    double* act_hist, double* prev_x, unsigned char* prev_x_valid, double* prev_alt_log,
    double* reward, unsigned char* term, unsigned char* trunc_,
    int nenv,
    double OX, double OY, double OZ, double OSLAT, double OCLAT, double OSLON, double OCLON,
    double dt, double min_alt, double max_time,
    double own_w, double dmg_scale, double altitude_scale, double shap_scale,
    double win_r, double loss_r,
    double timeout_win_r, double timeout_loss_r, double timeout_draw_r,
    int reward_mode, double alt_hunt_coef, int altitude_result_mode,
    double altitude_win_r, double altitude_loss_r) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nenv) return;
    int o = 2*e, p = 2*e + 1;

    // action history push(roll +1, row0=action)
    for (int t = 0; t < 2; t++) {
        int a = (t == 0) ? o : p;
        double* ah = act_hist + a*20;
        for (int k = 4; k >= 1; k--)
            for (int c = 0; c < 4; c++) ah[k*4 + c] = ah[(k-1)*4 + c];
        for (int c = 0; c < 4; c++) ah[c] = actions[a*4 + c];
    }

    double s9o[9], s9p[9];
    kin9(states + o*101, OX, OY, OZ, OSLAT, OCLAT, OSLON, OCLON, s9o);
    kin9(states + p*101, OX, OY, OZ, OSLAT, OCLAT, OSLON, OCLON, s9p);

    double dpos[3] = {s9p[0]-s9o[0], s9p[1]-s9o[1], s9p[2]-s9o[2]};
    double dist = vnorm(dpos);
    double r_ft = dist * M2FT;
    double ata_op = ata_deg(s9o, s9p);
    double ata_po = ata_deg(s9p, s9o);
    double t_old = t_sec[e];
    double rate_o = damage(r_ft, ata_op, t_old);
    double rate_p = damage(r_ft, ata_po, t_old);

    double hp_o_old = hp[o], hp_p_old = hp[p];
    double hp_o_new = hp_o_old - rate_p*dt; if (hp_o_new < 0.0) hp_o_new = 0.0;
    double hp_p_new = hp_p_old - rate_o*dt; if (hp_p_new < 0.0) hp_p_new = 0.0;
    double loss_o = hp_o_old - hp_o_new, loss_p = hp_p_old - hp_p_new;
    hp[o] = hp_o_new; hp[p] = hp_p_new;
    hp_loss[o] = loss_o; hp_loss[p] = loss_p;
    last_dmg_dealt[o] = rate_o; last_dmg_taken[o] = rate_p;
    last_dmg_dealt[p] = rate_p; last_dmg_taken[p] = rate_o;

    double spd_o = sqrt(s9o[6]*s9o[6] + s9o[7]*s9o[7] + s9o[8]*s9o[8]);
    double spd_p = sqrt(s9p[6]*s9p[6] + s9p[7]*s9p[7] + s9p[8]*s9p[8]);
    double fo = fuel[o] - FUEL_BURN*(spd_o/FUEL_REF)*dt; if (fo < 0.0) fo = 0.0; fuel[o] = fo;
    double fp = fuel[p] - FUEL_BURN*(spd_p/FUEL_REF)*dt; if (fp < 0.0) fp = 0.0; fuel[p] = fp;

    // pqr (SO3 log)
    for (int t = 0; t < 2; t++) {
        int a = (t == 0) ? o : p;
        const double* s9a = (t == 0) ? s9o : s9p;
        double Nprev[3][3], Ncurr[3][3];
        ned2body(prev_att[a*3+0], prev_att[a*3+1], prev_att[a*3+2], Nprev);
        ned2body(s9a[3], s9a[4], s9a[5], Ncurr);
        // r_delta = Nprev @ Ncurr^T
        double rd[3][3];
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
            double s=0; for (int k=0;k<3;k++) s += Nprev[i][k]*Ncurr[j][k]; rd[i][j]=s;
        }
        double lg[3];
        if (prev_valid[a]) { log_so3(rd, dt, lg); }
        else { lg[0]=lg[1]=lg[2]=0.0; }
        pqr[a*3+0]=lg[0]; pqr[a*3+1]=lg[1]; pqr[a*3+2]=lg[2];
        // 선가속도: NED 속도(body->NED, Ncurr^T@body_vel)의 step 차분. prev_valid 를 pqr
        // 과 공유하므로(둘 다 "직전 step 이 있었는가") 반드시 아래 prev_valid 갱신 전에 읽는다.
        double vel_ned[3];
        for (int i=0;i<3;i++) vel_ned[i] = Ncurr[0][i]*s9a[6] + Ncurr[1][i]*s9a[7] + Ncurr[2][i]*s9a[8];
        if (prev_valid[a]) {
            for (int i=0;i<3;i++) accel[a*3+i] = (vel_ned[i] - prev_vel_ned[a*3+i]) / dt;
        } else {
            accel[a*3+0]=accel[a*3+1]=accel[a*3+2]=0.0;
        }
        prev_vel_ned[a*3+0]=vel_ned[0]; prev_vel_ned[a*3+1]=vel_ned[1]; prev_vel_ned[a*3+2]=vel_ned[2];
        prev_att[a*3+0]=s9a[3]; prev_att[a*3+1]=s9a[4]; prev_att[a*3+2]=s9a[5];
        prev_valid[a]=1;
    }

    double t_new = t_old + dt; t_sec[e] = t_new;

    // 종료
    double alt_o = -s9o[2], alt_p = -s9p[2];
    int finite = 1;
    for (int k=0;k<9;k++){ if(!isfinite(s9o[k])||!isfinite(s9p[k])) finite=0; }
    unsigned char te = (alt_o < min_alt) || (alt_p < min_alt)
                    || (hp_o_new <= 0.0) || (hp_p_new <= 0.0) || (!finite);
    unsigned char tr = (!te) && (t_new >= max_time - TIME_LIMIT_TOLERANCE_SEC);
    term[e] = te; trunc_[e] = tr;

    // Standard: damage + final-safe geometry + terminal. Hunter: target log-altitude
    // progress replaces geometry. Neither mode has own low-altitude shaping.
    double geom_o=geometry_advantage(s9o,s9p,t_new), geom_p=-geom_o;
    double alt_log_o=altitude_log(alt_o), alt_log_p=altitude_log(alt_p);
    double prev_geom_o=prev_x_valid[o]?prev_x[o]:geom_o;
    double prev_geom_p=prev_x_valid[p]?prev_x[p]:geom_p;
    double geom_mult=shap_scale/SHAPING_BASE;
    double r_geom_o=geom_mult*(GEOM_BUDGET/GEOM_REF_SEC)*0.5*(prev_geom_o+geom_o)*dt;
    double r_geom_p=geom_mult*(GEOM_BUDGET/GEOM_REF_SEC)*0.5*(prev_geom_p+geom_p)*dt;
    if(reward_mode==1) {
        r_geom_o=prev_x_valid[p] ? alt_hunt_coef*(prev_alt_log[p]-alt_log_p) : 0.0;
        r_geom_p=prev_x_valid[o] ? alt_hunt_coef*(prev_alt_log[o]-alt_log_o) : 0.0;
    }
    prev_x[o]=geom_o; prev_x[p]=geom_p; prev_x_valid[o]=1; prev_x_valid[p]=1;
    prev_alt_log[o]=alt_log_o; prev_alt_log[p]=alt_log_p;

    double r_dam_o=(loss_p-loss_o*own_w)*dmg_scale;
    double r_dam_p=(loss_o-loss_p*own_w)*dmg_scale;
    // Extreme styles affect DAMAGE ONLY; terminal remaining HP still uses
    // the independent altitude_scale regardless of damage style/coefficient.
    if(reward_mode==2) { r_dam_o=loss_p*dmg_scale; r_dam_p=loss_o*dmg_scale; }
    else if(reward_mode==3) { r_dam_o=-(loss_o*own_w)*dmg_scale; r_dam_p=-(loss_p*own_w)*dmg_scale; }
    double r_term_o=0.0,r_term_p=0.0;
    if(te){
        if(altitude_result_mode) {
            // Same outcome predicate as online/evaluation scoring; simultaneous
            // deaths are a draw. Mode 2 additionally settles the AFTER-damage
            // remaining HP exactly once; mode 1 pays only the result.
            int own_dead=(alt_o<min_alt)||(hp_o_new<=0.0);
            int opp_dead=(alt_p<min_alt)||(hp_p_new<=0.0);
            r_term_o=(opp_dead&&!own_dead)?win_r:((own_dead&&!opp_dead)?loss_r:0.0);
            r_term_p=(own_dead&&!opp_dead)?win_r:((opp_dead&&!own_dead)?loss_r:0.0);
            if(alt_p<min_alt&&!own_dead) r_term_o=altitude_win_r;
            if(alt_o<min_alt&&!opp_dead) r_term_o=altitude_loss_r;
            if(alt_o<min_alt&&!opp_dead) r_term_p=altitude_win_r;
            if(alt_p<min_alt&&!own_dead) r_term_p=altitude_loss_r;
            if(altitude_result_mode==2) {
                if(alt_p<min_alt&&!own_dead) { r_term_o+=hp_p_new*altitude_scale; r_term_p-=hp_p_new*altitude_scale; }
                if(alt_o<min_alt&&!opp_dead) { r_term_o-=hp_o_new*altitude_scale; r_term_p+=hp_o_new*altitude_scale; }
            }
        } else {
        // 고도 종료는 HP 종료와 합산하지 않고 명시적 우선순위를 적용한다.
        if(alt_o<min_alt) r_term_o=-hp_o_new*altitude_scale;
        else if(alt_p<min_alt) r_term_o=hp_p_new*altitude_scale;
        else { if(hp_p_new<=0.0) r_term_o+=win_r; if(hp_o_new<=0.0) r_term_o+=loss_r; }
        if(alt_p<min_alt) r_term_p=-hp_p_new*altitude_scale;
        else if(alt_o<min_alt) r_term_p=hp_o_new*altitude_scale;
        else { if(hp_o_new<=0.0) r_term_p+=win_r; if(hp_p_new<=0.0) r_term_p+=loss_r; }
        }
    } else if(tr){
        r_term_o=(hp_o_new>hp_p_new)?timeout_win_r:((hp_o_new<hp_p_new)?timeout_loss_r:timeout_draw_r);
        r_term_p=(hp_p_new>hp_o_new)?timeout_win_r:((hp_p_new<hp_o_new)?timeout_loss_r:timeout_draw_r);
    }
    reward[o]=r_dam_o+r_geom_o+r_term_o;
    reward[p]=r_dam_p+r_geom_p+r_term_p;
}

extern "C" __global__ void build_obs_kernel(
    const double* states, const unsigned char* env_mask,
    const double* hp, const double* fuel, const double* t_sec,
    const double* pqr, const double* accel,
    const double* last_dmg_dealt, const double* last_dmg_taken,
    const double* act_hist, float* obs, int nac,
    double OX, double OY, double OZ, double OSLAT, double OCLAT, double OSLON, double OCLON,
    double* aux_features) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= nac) return;
    if (!env_mask[a >> 1]) return;  // autoreset: untouched lanes already hold terminal obs
    int par = a ^ 1;
    int e = a >> 1;
    double s9o[9], s9t[9];
    kin9(states + a*101, OX, OY, OZ, OSLAT, OCLAT, OSLON, OCLON, s9o);
    kin9(states + par*101, OX, OY, OZ, OSLAT, OCLAT, OSLON, OCLON, s9t);
    double pqr_o[3] = {pqr[a*3+0], pqr[a*3+1], pqr[a*3+2]};
    double pqr_p[3] = {pqr[par*3+0], pqr[par*3+1], pqr[par*3+2]};
    double accel_o[3] = {accel[a*3+0], accel[a*3+1], accel[a*3+2]};
    double accel_p[3] = {accel[par*3+0], accel[par*3+1], accel[par*3+2]};
    build_obs_one(s9o, s9t, hp[a], hp[par], fuel[a], fuel[par],
                  pqr_o, pqr_p, accel_o, accel_p, last_dmg_dealt[a], last_dmg_taken[a],
                  t_sec[e], act_hist + a*20, obs + a*214,
                  (aux_features && !(a & 1)) ? aux_features + e*21 : nullptr);
}
