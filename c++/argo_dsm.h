/*
 * Copyright (c) 2019-2020, I. Anevlavis & K. Palaiodimos
 * All rights reserved.
 */


#include <mpi.h>
#include "error.h"
#include "argo.hpp"

#define SYS ARGO_Sys


void Sys::Init(const std::size_t& nnodes)
{
    std::size_t init_glmem               =    sizeof(double) * nnodes                                            +   // norm_ptr
                                              sizeof(double) * nnodes                                            +   // >>
                                              sizeof(double) * BPMF_NUMLATENT * nnodes                           +   // sum_ptr
                                              sizeof(double) * BPMF_NUMLATENT * nnodes                           +   // >>
                                              sizeof(double) * BPMF_NUMLATENT * BPMF_NUMLATENT * nnodes          +   // cov_ptr
                                              sizeof(double) * BPMF_NUMLATENT * BPMF_NUMLATENT * nnodes          +   // >>
                                              sizeof(double) * BPMF_NUMLATENT * Sys::nmovies                     +   // items_ptr
                                              sizeof(double) * BPMF_NUMLATENT * Sys::nusers;                         // >>

    init_glmem += (Sys::odirname.size()) ?    sizeof(double) * BPMF_NUMLATENT * Sys::nmovies                     +   // aggrMu_ptr
                                              sizeof(double) * BPMF_NUMLATENT * Sys::nusers                      +   // >>
                                              sizeof(double) * BPMF_NUMLATENT * BPMF_NUMLATENT * Sys::nmovies    +   // aggrLambda_ptr
                                              sizeof(double) * BPMF_NUMLATENT * BPMF_NUMLATENT * Sys::nusers         // >>
                                         :    0;

    init_glmem += (Sys::odirname.size()) ?    8 * 4 * 1024UL                // something bigger
                                         :    8 * 4 * 1024UL;               // something smaller

    const std::size_t init_cache = init_glmem;

    THROWERROR_ASSERT(init_glmem <  22 * 1024 * 1024 * 1024UL * nnodes);    // for Jason
    THROWERROR_ASSERT(init_glmem < 126 * 1024 * 1024 * 1024UL * nnodes);    // for Rackham
    
    argo::init(init_glmem, init_cache);

    Sys::procid = argo::node_id();
    Sys::nprocs = argo::number_of_nodes();

    THROWERROR_ASSERT((int)nnodes == Sys::nprocs);

    std::cout << "Alloc is : "  << init_glmem   <<
                 ", nnodes : "  << nnodes       <<
                 ", nusers : "  << Sys::nusers  <<
                 ", nmovies : " << Sys::nmovies <<
                 ", procid : "  << Sys::procid  <<
                 ", out of : "  << Sys::nprocs  << 
    std::endl;
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
    // #############################################################
    // c'tor
    // #############################################################
    ARGO_Sys(std::string name, std::string fname, std::string probename) : Sys(name, fname, probename) {}
    ARGO_Sys(std::string name, const SparseMatrixD &M, const SparseMatrixD &P) : Sys(name, M, P) {}
    
    ~ARGO_Sys();

    virtual void sample_hp();
    virtual void sample(Sys &in);
    virtual void send_item(int) {}
    virtual void alloc_and_init();

    void init_after();
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
    // #############################################################
    // Movies : M x K (V matrix) | Users : N x K (U matrix)
    // #############################################################
    norm_ptr  = argo::conew_array<double>(ARGO_Sys::nprocs);
    sum_ptr   = argo::conew_array<double>(num_latent * ARGO_Sys::nprocs);
    cov_ptr   = argo::conew_array<double>(num_latent * num_latent * ARGO_Sys::nprocs);
    items_ptr = argo::conew_array<double>(num_latent * num());

    if (Sys::odirname.size()) {
        aggrMu_ptr     = argo::conew_array<double>(num_latent * num());
        aggrLambda_ptr = argo::conew_array<double>(num_latent * num_latent * num());
    }
    
    { BPMF_COUNTER("init"); init(); }
}


void ARGO_Sys::init_after()
{
    // #########################################################
    // Each process initializes it's own chunk (by size) to 0
    // #########################################################
    {
        BPMF_COUNTER("init");
        
        const VectorNd zero = VectorNd::Zero();

        // -----------------------------------------------------
        norm(Sys::procid) = 0;
        // -----------------------------------------------------

        // -----------------------------------------------------
        sum_map().col(Sys::procid) = zero;
        // -----------------------------------------------------

        // -----------------------------------------------------
        int chunk = num_latent;
        int data_begin = Sys::procid * chunk;
        int data_end = (Sys::procid != Sys::nprocs - 1) ? data_begin + chunk : Sys::nprocs * num_latent;

        for (int i = data_begin; i < data_end; ++i)
            cov_map().col(i) = zero;
        // -----------------------------------------------------

        // -----------------------------------------------------
        for (int i = from(); i < to(); ++i)
            items().col(i) = zero;
        // -----------------------------------------------------

        Sys::sync(); // Added for debugging
    }
}


void ARGO_Sys::argo_bcast()
{
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
