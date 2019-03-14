#include <math.h>

void undo_thing3d_m(double *in, double *out, int xn, int yn, int zn, int scalefac, double fill_value) {
	for (int z=0; z<zn; z++) {
		int zidx = z * xn * yn;
		for (int x=0; x<xn; x++) {
			for (int y=0; y<yn; y++) {
				int out_idx = zidx + x*yn + y;
				int block_idx = zidx * scalefac * scalefac + x*yn*scalefac*scalefac + y*scalefac;

				double out_val = 0.0;
				int vcells = 0;

				for (int xo=0; xo < scalefac; xo++) {
					for (int yo=0; yo < scalefac; yo++) {
						int in_idx = block_idx + xo * scalefac * yn + yo;
						double in_val = in[in_idx];
						if (in_val != fill_value) {
							out_val += in[in_idx];
							vcells ++;
						}
					}
				}
				if (vcells > 0) {
					out[out_idx] = out_val / vcells;//(scalefac * scalefac);
				} else {
					out[out_idx] = fill_value;
				}
			}
		}
	}
}

template <typename T>
void do_thing3d(T *in, T *out, int xn, int yn, int zn, int scalefac) {
	for (int z=0; z<zn; z++) {
		int zidx = z * xn * yn;
		for (int x=0; x<xn; x++) {
		for (int y=0; y<yn; y++) {			
				int in_idx = zidx + x*yn + y;
				T in_val = in[in_idx];
				int block_idx = zidx * scalefac * scalefac + x*yn*scalefac*scalefac + y*scalefac;
				for (int xo=0; xo < scalefac; xo++) {
				for (int yo=0; yo < scalefac; yo++) {
						int out_idx = block_idx + xo * scalefac * yn + yo;
						out[out_idx] = in_val;
					}
				}
			}
		}
	}
}

template <typename T>
void undo_thing3d(T *in, T *out, int xn, int yn, int zn, int scalefac) {
	for (int z=0; z<zn; z++) {
		int zidx = z * xn * yn;
		for (int x=0; x<xn; x++) {
			for (int y=0; y<yn; y++) {
				int out_idx = zidx + x*yn + y;
				int block_idx = zidx * scalefac * scalefac + x*yn*scalefac*scalefac + y*scalefac;

				T out_val = 0.0;

				for (int xo=0; xo < scalefac; xo++) {
					for (int yo=0; yo < scalefac; yo++) {
						int in_idx = block_idx + xo * scalefac * yn + yo;
						out_val += in[in_idx];
					}
				}
				out[out_idx] = out_val / (scalefac * scalefac);
			}
		}
	}
}

template <typename T>
void undo_thing3d_m(T *in, T *out, int xn, int yn, int zn, int scalefac) {
	for (int z=0; z<zn; z++) {
		int zidx = z * xn * yn;
		for (int x=0; x<xn; x++) {
			for (int y=0; y<yn; y++) {
				int out_idx = zidx + x*yn + y;
				int block_idx = zidx * scalefac * scalefac + x*yn*scalefac*scalefac + y*scalefac;

				T out_val = 0.0;
				int vcells = 0;

				for (int xo=0; xo < scalefac; xo++) {
					for (int yo=0; yo < scalefac; yo++) {
						int in_idx = block_idx + xo * scalefac * yn + yo;
						T in_val = in[in_idx];
						if (!isnan(in_val)) {
							out_val += in[in_idx];
							vcells ++;
						}
					}
				}
				if (vcells > 0) {
					out[out_idx] = out_val / vcells;//(scalefac * scalefac);
				} else {
					out[out_idx] = NAN;
				}
			}
		}
	}
}

extern "C" {

	void do_thing3d_d(double *in, double *out, int xn, int yn, int zn, int scalefac) {
		do_thing3d<double>(in,out,xn,yn,zn,scalefac);
	}

	void do_thing3d_f(float *in, float *out, int xn, int yn, int zn, int scalefac) {
		do_thing3d<float>(in,out,xn,yn,zn,scalefac);
	}
	void undo_thing3d_d(double *in, double *out, int xn, int yn, int zn, int scalefac) {
		undo_thing3d<double>(in,out,xn,yn,zn,scalefac);
	}

	void undo_thing3d_f(float *in, float *out, int xn, int yn, int zn, int scalefac) {
		undo_thing3d<float>(in,out,xn,yn,zn,scalefac);
	}

	void undo_thing3d_m_d(double *in, double *out, int xn, int yn, int zn, int scalefac) {
		undo_thing3d_m<double>(in,out,xn,yn,zn,scalefac);
	}

	void undo_thing3d_m_f(float *in, float *out, int xn, int yn, int zn, int scalefac) {
		undo_thing3d_m<float>(in,out,xn,yn,zn,scalefac);
	}
}

