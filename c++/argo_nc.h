/*
 * Copyright (c) 2019-2020, I. Anevlavis & K. Palaiodimos
 * All rights reserved.
 */

#include <mpi.h>
#include "argo.hpp"

///////////////////////////////////////
#define NNODES 8

#define BPMF_NUMOVIES 1050096UL
#define BPMF_NUMUSERS 1107944UL
///////////////////////////////////////

#define SYS ARGO_Sys

/*
void Sys::Init(const unsigned long long& init_num_latent, const unsigned long long& init_num_users, const unsigned long long& init_num_movies)
{   
    unsigned long long init_sum        =    sizeof(double) * init_num_latent * init_num_movies                   +  // items_ptr
                                            sizeof(double) * init_num_latent * init_num_users                    +  // >>
                                            sizeof(double) * init_num_latent * NNODES                            +  // sum_ptr
                                            sizeof(double) * init_num_latent * NNODES                            +  // >>
                                            sizeof(double) * init_num_latent * init_num_latent * NNODES          +  // cov_ptr
                                            sizeof(double) * init_num_latent * init_num_latent * NNODES          +  // >>
                                            sizeof(double) * NNODES                                              +  // norm_ptr
                                            sizeof(double) * NNODES;                                                // >>

    init_sum += (Sys::odirname.size()) ?    sizeof(double) * init_num_latent * init_num_movies                   +  // aggrMu_ptr
                                            sizeof(double) * init_num_latent * init_num_users                    +  // >>
                                            sizeof(double) * init_num_latent * init_num_latent * init_num_movies +  // aggrLambda_ptr
                                            sizeof(double) * init_num_latent * init_num_latent * init_num_users     // >>
                                       :    0;

    //init_sum += (Sys::odirname.size()) ?    2 * 4 * 1024ULL
    //                                   :    10 * 1024 * 1024 * 1024UL; //100 * 4 * 1024ULL;

    init_sum += (1503472 * init_sum) / 62644672;

    assert(init_sum < 23622320128 * NNODES);
    
    argo::init(init_sum);
    
    Sys::procid = argo::node_id();
    Sys::nprocs = argo::number_of_nodes();

    std::cout << "Allocated size is : " << init_sum << ", process : " << Sys::procid << " out of " << Sys::nprocs << std::endl;
}
*/

void Sys::Init()
{
    std::size_t init_sum               =    sizeof(double) * BPMF_NUMLATENT * BPMF_NUMOVIES                  +  // items_ptr
                                            sizeof(double) * BPMF_NUMLATENT * BPMF_NUMUSERS                  +  // >>
                                            sizeof(double) * BPMF_NUMLATENT * NNODES                         +  // sum_ptr
                                            sizeof(double) * BPMF_NUMLATENT * NNODES                         +  // >>
                                            sizeof(double) * BPMF_NUMLATENT * BPMF_NUMLATENT * NNODES        +  // cov_ptr
                                            sizeof(double) * BPMF_NUMLATENT * BPMF_NUMLATENT * NNODES        +  // >>
                                            sizeof(double) * NNODES                                          +  // norm_ptr
                                            sizeof(double) * NNODES;                                            // >>

    init_sum += (Sys::odirname.size()) ?    sizeof(double) * BPMF_NUMLATENT * BPMF_NUMOVIES                  +  // aggrMu_ptr
                                            sizeof(double) * BPMF_NUMLATENT * BPMF_NUMUSERS                  +  // >>
                                            sizeof(double) * BPMF_NUMLATENT * BPMF_NUMLATENT * BPMF_NUMOVIES +  // aggrLambda_ptr
                                            sizeof(double) * BPMF_NUMLATENT * BPMF_NUMLATENT * BPMF_NUMUSERS    // >>
                                       :    0;

    init_sum += (Sys::odirname.size()) ?    2 * 4 * 1024ULL
                                       :    4 * 4 * 1024ULL;

    assert(init_sum < 23622320128 * NNODES);
    
    argo::init(init_sum);

    Sys::procid = argo::node_id();
    Sys::nprocs = argo::number_of_nodes();

    std::cout << "Allocated size is : " << init_sum << ", process : " << Sys::procid << " out of " << Sys::nprocs << std::endl;
}

void Sys::Finalize()
{
#ifdef BPMF_PROFILING
    perf_data.print();
#endif
    argo::finalize();
}

void Sys::sync()
{
    argo::barrier();
}

void Sys::Abort(int err)
{
    MPI_Abort(MPI_COMM_WORLD, err);
}

struct ARGO_Sys : public Sys
{
    //-- c'tor
    ARGO_Sys(std::string name, std::string fname, std::string probename) : Sys(name, fname, probename) {}
    ARGO_Sys(std::string name, const SparseMatrixD &M, const SparseMatrixD &P) : Sys(name, M, P) {}
    
    ~ARGO_Sys();

    virtual void sample_hp();
    virtual void sample(Sys &in);
    virtual void send_item(int) {}
    virtual void alloc_and_init();

    void argo_bcast();
};

ARGO_Sys::~ARGO_Sys()
{
    argo::codelete_array(items_ptr);
    argo::codelete_array(sum_ptr);
    argo::codelete_array(cov_ptr);
    argo::codelete_array(norm_ptr);

    if (Sys::odirname.size()) {
        argo::codelete_array(aggrMu_ptr);
        argo::codelete_array(aggrLambda_ptr);
    }
}

void ARGO_Sys::alloc_and_init()
{
    // Movies : M x K (V matrix) | Users : N x K (U matrix)
    sum_ptr   = argo::conew_array<double>(num_latent * ARGO_Sys::nprocs);
    cov_ptr   = argo::conew_array<double>(num_latent * num_latent * ARGO_Sys::nprocs);
    norm_ptr  = argo::conew_array<double>(ARGO_Sys::nprocs);
    items_ptr = argo::conew_array<double>(num_latent * num());

    if (Sys::odirname.size()) {
        aggrMu_ptr     = argo::conew_array<double>(num_latent * num());
        aggrLambda_ptr = argo::conew_array<double>(num_latent * num_latent * num());
    }
    
    { BPMF_COUNTER("init"); init(); }
}

void ARGO_Sys::argo_bcast()
{
    // Take note, waste for NNODES 1
#pragma omp parallel for schedule(guided)
    for (int i = from(); i < to(); ++i)
    {
#pragma omp task
        {
            aggrMus().col(i) = aggrMu.col(i);
            aggrLambdas().col(i) = aggrLambda.col(i);
        }
    }

    { BPMF_COUNTER("sync_2");  Sys::sync();     }
}

void ARGO_Sys::sample(Sys &in)
{
    { BPMF_COUNTER("compute"); Sys::sample(in); }
    
    { BPMF_COUNTER("sync_1");  Sys::sync();     }
}

void ARGO_Sys::sample_hp()
{
    { /*BPMF_COUNTER("compute");*/ Sys::sample_hp(); }

    { BPMF_COUNTER("sync_0");  Sys::sync();     }
}
