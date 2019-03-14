typedef struct {
    double *restrict s0, *restrict ss, *restrict sd, *restrict mleaf;
} HRUState;

typedef struct {
    double *restrict sg, *restrict sr;
    HRUState hru[2];
} States;

typedef struct {
    const double *ne, *height, *hypsperc;
} Hypsometry;

//ATL_BEGIN <STRUCT_DEFS>

#ifdef _WIN32
#define EXPORT_EXTENSION extern __declspec(dllexport)
#else
#define EXPORT_EXTENSION extern
#endif

EXPORT_EXTENSION
void awral(Forcing inputs, Outputs outputs, States states, 
           Parameters params, Spatial spatial, Hypsometry hypso, HRUParameters *hruparams, HRUSpatial *hruspatial,
           int timesteps, int cells);
