#include "awral.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

const int HYPS_LEN = 20; // +++ Should be supplied as input value; 20 hardcoded in other places

void copycells(const double *restrict in, double *restrict out, int len) {
    for (int i =0; i<len; i++){
        out[i] = in[i];
    }
}

void build_hypso(double *restrict sgtemp, const double *restrict height, const double *restrict ne, int cells) {
    for (int c = 0; c < cells; c++) {
        for (int l = 0; l < 20; l++) {
            int idx = c * HYPS_LEN + l;
            sgtemp[idx] = height[idx] * ne[c] * 1000.0;
        }
    }
}

__inline__ double hyps_fsat(double *restrict sgtemp, const double *restrict hypsfsat, double sg, double offset) {

    double sgmin = sgtemp[0] + offset;
    double sg_eff = sgmin+sg;
    if (sg + offset <= 0.) {
        return(0);
    } else if (sg_eff >= sgtemp[HYPS_LEN-1]) {
        return(1.0);
    } else {
        int id0 = -1;
        for (int i=0;i<HYPS_LEN;i++) {
            if (sg_eff >= sgtemp[i]) {
                id0++;
            } else {
                break;
            }
        }
        int id1 = id0+1;

        double srange = sgtemp[id1]-sgtemp[id0];
        double hdiff = hypsfsat[id1]-hypsfsat[id0];

        return (hypsfsat[id0]+hdiff*((sg_eff-sgtemp[id0])/srange));
    }

}

#define HRU_SUM(HRUVAL) hru_sum((double *)HRUVAL,fhru,hidx);

__inline__ double hru_sum(double *hru_data,double fhru[2],int hidx[2]) {
    return (fhru[0]*hru_data[hidx[0]]+fhru[1]*hru_data[hidx[1]]);
}

__inline__ void calc_soil_flows(double *restrict s, double *restrict i, double *restrict e, double *restrict drain, double *restrict iflow, const double smax, const double ksat, const double rh, const double km) {
    if ((*s + *i) <= *e) {
        *e = *s + *i;
        *s = 0.0;
        *drain = 0.0;
        *iflow = 0.0;
    } else {
        double a = ksat / (smax*smax);
        const double b = 1.0;
        double c = -(*s + *i - *e);
        if ((smax - *s + ksat) <= (*i - *e)) {
            *drain = (1 - rh) * ksat;
            *iflow = (rh * ksat) + (*s - smax - ksat + *i - *e);
            *s = smax;
        } else {
            *s = (-b + (pow(b * b - 4. * a * c,0.5))) / (2. * a);
            if (*s < smax) {
                double pfull = (*s/smax);
                *drain = (1 - rh) * ksat * pfull * pfull;
                *iflow = rh * ksat * pfull * pfull;
            } else {
                *drain = (1 - rh) * ksat;
                *iflow = (rh * ksat) + (-c - smax -ksat);
                *s = smax;
            }
        }
    }
}

void awral(Forcing inputs, Outputs outputs, States states, 
           Parameters params, Spatial spatial, Hypsometry hypso, HRUParameters *hruparams, HRUSpatial *hruspatial,
           int timesteps, int cells) {

    // +++ These could use the final_states arrays and avoid the extra mallocs...
    // More generally, could have user supplied arrays for all mallocs...
    // Declare States
    //double *sg_ = malloc(cells*sizeof(double));
    //double *sr_ = malloc(cells*sizeof(double));

    // HRU States

    //double *s0_ = malloc(cells*2*sizeof(double));
    //double *ss_ = malloc(cells*2*sizeof(double));
    //double *sd_ = malloc(cells*2*sizeof(double));
    //double *mleaf_ = malloc(cells*2*sizeof(double));

    // HRU Values exported to non-HRU regions

    double *dd_ = malloc(cells*2*sizeof(double));
    double *eg_ = malloc(cells*2*sizeof(double));
    double *y_ = malloc(cells*2*sizeof(double));
    double *qif_ = malloc(cells*2*sizeof(double));
    double *qr_ = malloc(cells*2*sizeof(double));

    // Copy initial states
    //copycells(initial_states.sg,sg_,cells);
    //copycells(initial_states.sr,sr_,cells);

    //#pragma ivdep
    //for (int hru=0; hru<2; hru++){
     //   copycells(initial_states.hru[hru].s0,&s0_[hru*cells],cells);
    //    copycells(initial_states.hru[hru].ss,&ss_[hru*cells],cells);
    //    copycells(initial_states.hru[hru].sd,&sd_[hru*cells],cells);
    //    copycells(initial_states.hru[hru].mleaf,&mleaf_[hru*cells],cells);
    //}

    const double stefbolz  = 5.67e-8;

    double *sgtemp_ = malloc(cells*20*sizeof(double));

    build_hypso(sgtemp_,hypso.height,hypso.ne,cells);

    double *eff_rd = malloc(cells*2*sizeof(double));//[2][cells];

    for (int hru=0; hru<2; hru++){
        for (int c=0; c<cells; c++) {
            eff_rd[(hru*cells)+c] = hruparams[hru].rd * 1000.0 * hypso.ne[c];
        }
    }

    double *fsat_ = malloc(cells*sizeof(double));//[cells];
    double *fegt_ = malloc(cells*sizeof(double));//[cells];

    // Load scalar constants
    double kr_coeff = params.kr_coeff;
    double pair = params.pair;
    double pt = params.pt;
    double slope_coeff = params.slope_coeff;


    for (int ts=0; ts<timesteps; ts++) {

        // Hypso
        //double fsat_[cells];

        for (int c=0; c<cells; c++) {
            fsat_[c] = hyps_fsat(&sgtemp_[c*20],hypso.hypsperc,states.sg[c],0.0);
            int idx = cells*ts + c;

            //Zero fill fhru-averaged outputs
            outputs.e0[idx] = 0.0;
            outputs.etot[idx] = 0.0;
            outputs.dd[idx] = 0.0;
            outputs.s0[idx] = 0.0;
            outputs.ss[idx] = 0.0;
            outputs.sd[idx] = 0.0;
            outputs.ifs[idx] = 0.0;
        }

        // HRU loop
        for (int hru=0; hru<2; hru++) {
            HRUParameters hrup = hruparams[hru];
            HRUSpatial hrus = hruspatial[hru];

            //double fegt_[cells];

            for (int c=0; c<cells; c++) {
                fegt_[c] = hyps_fsat(&sgtemp_[c*20],hypso.hypsperc,states.sg[c],eff_rd[(hru*cells)+c]);
            }

            double alb_dry = hrup.alb_dry;
            double alb_wet = hrup.alb_wet;
            double cgsmax = hrup.cgsmax;
            double er_frac_ref = hrup.er_frac_ref;
            double fsoilemax = hrup.fsoilemax;
            double lairef = hrup.lairef;
            double rd = hrup.rd;
            double s_sls = hrup.s_sls;
            double sla = hrup.sla;
            double tgrow = hrup.tgrow;
            double tsenc = hrup.tsenc;
            double ud0 = hrup.ud0;
            double us0 = hrup.us0;
            double vc = hrup.vc;
            double w0lime = hrup.w0lime;
            double w0ref_alb = hrup.w0ref_alb;
            double wdlimu = hrup.wdlimu;
            double wslimu = hrup.wslimu;


            #pragma ivdep
            #pragma vector always
            for (int c=0; c<cells; c++) {
                int idx = cells*ts + c;
                int hru_cidx = hru*cells+c;

                //double s0 = s0_[hru][c];
                //double ss = ss_[hru][c];
                //double sd = sd_[hru][c];
                //double mleaf = mleaf_[hru][c];
                double s0 = states.hru[hru].s0[c];
                double ss = states.hru[hru].ss[c];
                double sd = states.hru[hru].sd[c];
                double mleaf = states.hru[hru].mleaf[c];

                //

                //Load spatial data
                double k0sat = spatial.k0sat[c];
                double k_gw = spatial.k_gw[c];
                double k_rout = spatial.k_rout[c];
                double kdsat = spatial.kdsat[c];
                double kr_0s = spatial.kr_0s[c];
                double kr_sd = spatial.kr_sd[c];
                double kssat = spatial.kssat[c];
                double prefr = spatial.prefr[c];
                double s0max = spatial.s0max[c];
                double sdmax = spatial.sdmax[c];
                double slope = spatial.slope[c];
                double ssmax = spatial.ssmax[c];
                double ne = spatial.ne[c];

                //Load HRU spatial data
                double fhru = hrus.fhru[c];
                double hveg = hrus.hveg[c];
                double laimax = hrus.laimax[c];

                //Load forcing data
                float avpt = inputs.avpt[idx];
                float radcskyt = inputs.radcskyt[idx];
                float rgt = inputs.rgt[idx];
                float tat = inputs.tat[idx];
                float u2t = inputs.u2t[idx];

                // +++ 
                // Hideous alias stuff due to variable names not matching
                // Need to decide on new standard symbols/abbreviations for forcing/climate data

                double ta = tat;
                double pg = pt;
                double rg = rgt;
                double pe = avpt;
                double u2 = u2t;
                double radclearsky = radcskyt;

                double lai = sla * mleaf;
                double fveg = 1.0 - exp(-lai / lairef);
                double fsoil = 1.0 - fveg;

                double w0 = s0 / s0max;
                double ws = ss / ssmax;
                double wd = sd / sdmax;

                double fsat = fsat_[c];
                double fegt = fegt_[c];

                double pes = 610.8 * exp(17.27 * ta / (237.3 + ta));
                double frh = pe / pes;

                double keps = 1.4e-3 * ((ta / 187.0) * (ta / 187.0) + ta / 107.0 + 1.0) * (6.36 * pair + pe) / pes;

                double delta = 4217.457 / ((240.97 + ta)*(240.97 + ta)) * pes;
                double gamma = 0.000646 * pair * (1.0 + 0.000946 * ta);
                double lambda = 2.501 - (0.002361 * ta);

                double rgeff = rg;

                double alb_veg = 0.452 * vc;
                double alb_soil = alb_wet + (alb_dry - alb_wet) * exp(-w0 / w0ref_alb);
                double alb = fveg * alb_veg + fsoil * alb_soil;
                double rsn = (1.0 - alb) * rgeff;

                double tkelv = ta + 273.15;
                double rlin = stefbolz * pow(tkelv,4.0) * (1.0 - (1.0 - 0.65 * pow((pe / tkelv),0.14)) * (1.35 * rgeff / radclearsky - 0.35));
                double rlout = 1.0 * stefbolz * pow(tkelv,4);
                double rln = (rlin - rlout) * 0.0864;
                double rneff = rsn + rln;

                double fh = log(813. / hveg - 5.45);
                double ku2 = 0.305 / (fh * (fh + 2.3));
                double ga = ku2*u2;

                //  Potential evaporation
                //New for v4 (different to original in Van Dijk (2010)) 
                double e0 = (delta * rneff + 6.43 / 1000.0 * gamma * (pes - pe) * (1.0 + 0.536 * u2)) / (delta + gamma) / lambda;

                e0 = fmax(e0,0.);
                double usmax = us0*fmin(ws/wslimu,1.0);
                double udmax = ud0*fmin(wd/wdlimu,1.0);
                double u0 = fmax(usmax,udmax);

                double gsmax = cgsmax * vc;
                double gs = fveg * gsmax;
                double ft = 1.0 / (1.0 + (keps / (1.0 + keps)) * ga / gs);
                double etmax = ft*e0;

                double et = fmin(u0,etmax);

                double us, ud;

                if (u0 > 0.0) {
                    us = fmin( (usmax/(usmax+udmax)) * et,ss-0.01);
                    ud = fmin( (udmax/(usmax+udmax)) * et,sd-0.01);
                } else {
                    us = 0.0;
                    ud = 0.0; 
                }

                et = us + ud;

                // Remaining ET
                double et_rem = e0-et;

                double fsoile = fsoilemax * fmin(w0/w0lime,1.);
                double es = (1.0 - fsat)*fsoile*(et_rem);
                double eg = fsat*fsoilemax*(et_rem);

                double sveg = s_sls * lai;
                double fer = er_frac_ref * fveg;
                double pwet = -log(1.0 - fer/fveg) * sveg / fer;

                // +++
                // Need to limit this and update remaining ET
                double ei;

                if (pg<pwet) {
                    ei = fveg * pg;
                } else {
                    ei = fveg * pwet + fer * (pg-pwet);
                }

                double pn = fmax(0.0,pg-ei);
                double rhof = (1.0 - fsat)*(pn - (prefr*tanh(pn/prefr)));
                double rsof = fsat * pn;
                double qr = rhof + rsof;
                double i = pn - qr;

                //+++ Purely spatial, should be in spatial_mapping
                double slope_eff = slope_coeff * slope;

                double rh_0s = tanh(slope_eff*w0)*tanh(kr_coeff*(kr_0s-1.0)*w0);
                double rh_sd = tanh(slope_eff*ws)*tanh(kr_coeff*(kr_sd-1.0)*ws);
                double rh_dd = 0.;

                double km_s0 = pow((k0sat * kssat),0.5);
                double km_ss = pow((kssat * kdsat),0.5);
                double km_sd = kdsat;

                double d0, if0, ds, ifs, dd, ifd;

                calc_soil_flows(&s0,&i,&es,&d0,&if0,s0max,k0sat, rh_0s, km_s0);
                calc_soil_flows(&ss,&d0,&us,&ds,&ifs,ssmax,kssat, rh_sd, km_ss);
                calc_soil_flows(&sd,&ds,&ud,&dd,&ifd,sdmax,kdsat, rh_dd, km_sd);

                et = us + ud;

                double wz = fmax(0.01, sd/sdmax);

                double y;

                // +++
                // Need to limit this and update remaining ET
                if ((fegt-fsat) > 0.0) {
                    y = (fegt-fsat) * fsoilemax * et_rem;
                } else {
                    y = 0.0;
                }

                double fvmax = 1. - exp(-fmax(laimax,0.0027777778450399637)/lairef);

                double fveq = (1.0 / fmax((e0/u0)-1.0,1e-3)) * (keps/(1.0+keps))*(ga/gsmax);
                fveq = fmin(fveq,fvmax);

                double dmleaf = -log(1.0 - fveq) * lairef / sla - mleaf;
                double mleafnet = 0.0;

                if (dmleaf > 0.) {
                    mleafnet = dmleaf/tgrow;
                } else if (dmleaf < 0.) {
                    mleafnet = dmleaf/tsenc;
                }
                
                mleaf += mleafnet;

                states.hru[hru].s0[c] = s0;
                states.hru[hru].ss[c] = ss;
                states.hru[hru].sd[c] = sd;
                states.hru[hru].mleaf[c] = mleaf;

                dd_[hru_cidx] = dd;
                eg_[hru_cidx] = eg;
                y_[hru_cidx] = y;
                qif_[hru_cidx] = if0 + ifs + ifd;
                qr_[hru_cidx] = qr;

                double ee = es + eg + ei + y;
                double etot = ee + et;

                //Template library will generate HRU specific outputs
                outputs.hru[hru].s0[idx] = s0;
                outputs.hru[hru].ss[idx] = ss;
                outputs.hru[hru].sd[idx] = sd;
                outputs.hru[hru].mleaf[idx] = mleaf;
                outputs.hru[hru].ifs[idx] = ifs;

                //Templated combined (FHRU-weighted) outputs
                outputs.e0[idx] += e0*fhru;
                outputs.etot[idx] += etot*fhru;
                outputs.dd[idx] += dd*fhru;
                outputs.s0[idx] += s0*fhru;
                outputs.ss[idx] += ss*fhru;
                outputs.sd[idx] += sd*fhru;
                outputs.ifs[idx] += ifs*fhru;


            } // end cell loop
        } // end HRU loop

        // Post HRU
        #pragma ivdep
        #pragma vector always
        for (int c=0; c<cells; c++) {
            double k0sat = spatial.k0sat[c];
            double k_gw = spatial.k_gw[c];
            double k_rout = spatial.k_rout[c];
            double kdsat = spatial.kdsat[c];
            double kr_0s = spatial.kr_0s[c];
            double kr_sd = spatial.kr_sd[c];
            double kssat = spatial.kssat[c];
            double prefr = spatial.prefr[c];
            double s0max = spatial.s0max[c];
            double sdmax = spatial.sdmax[c];
            double slope = spatial.slope[c];
            double ssmax = spatial.ssmax[c];
            double ne = spatial.ne[c];

            int hidx[2];
            hidx[0] = c;
            hidx[1] = cells + c;

            double fhru[2];
            fhru[0] = hruspatial[0].fhru[c];
            fhru[1] = hruspatial[1].fhru[c];

            double dd = HRU_SUM(dd_);
            double eg = HRU_SUM(eg_);
            double y = HRU_SUM(y_);
            double qif = HRU_SUM(qif_);
            double qr = HRU_SUM(qr_);

            double sg = states.sg[c];
            double sr = states.sr[c];

            sg = sg + dd;
            double qg = (1.0 - exp(-k_gw)) * fmax(0.0,sg);
            sg = sg - qg - eg - y;

            // calculate groundwater levels as AHD

            double gw_level = hypso.height[c*HYPS_LEN] + sg/(hypso.ne[c] * 1000.0);

            sr = fmax(0.0,(sr + qr + qg + qif));

            double kd = (1.0 - exp(-k_rout));

            double qtot = fmin(sr,(sr * kd));
            sr = sr - qtot;

            states.sr[c] = sr;
            states.sg[c] = sg;

            int idx = cells*ts + c;

            //Cell level template outputs
            outputs.qtot[idx] = qtot;
            outputs.sr[idx] = sr;
            outputs.sg[idx] = sg;
        }
    }
        // Copy initial states
    //copycells(sg_,final_states.sg,cells);
    //copycells(sr_,final_states.sr,cells);

    //#pragma ivdep
    //for (int hru=0; hru<2; hru++){
    //    copycells(s0_+hru*cells,final_states.hru[hru].s0,cells);
    //    copycells(ss_+hru*cells,final_states.hru[hru].ss,cells);
    //    copycells(sd_+hru*cells,final_states.hru[hru].sd,cells);
    //   copycells(mleaf_+hru*cells,final_states.hru[hru].mleaf,cells);
    //}

    //free(sg_);
    //free(sr_);

    //free(s0_);
    //free(ss_);
    //free(sd_);
    //free(mleaf_);
    free(dd_);
    free(eg_);
    free(y_);
    free(qif_);
    free(qr_);

    free(sgtemp_);
    free(eff_rd);

    free(fsat_);
    free(fegt_);


}


