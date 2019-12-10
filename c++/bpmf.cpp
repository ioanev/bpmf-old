/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */


#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include <unistd.h>

#include "io.h"
#include "bpmf.h"

using namespace std;
using namespace Eigen;

#ifdef BPMF_HYBRID_COMM
#define BPMF_GPI_COMM
#define BPMF_MPI_COMM
#endif

#ifdef BPMF_GPI_COMM
#include "gaspi.h"
#elif defined(BPMF_ARGO_COMM)
#include "argo_dsm.h"
#elif defined(BPMF_MPI_PUT_COMM)
#define BPMF_MPI_COMM
#include "mpi_put.h"
#elif defined(BPMF_MPI_BCAST_COMM)
#define BPMF_MPI_COMM
#include "mpi_bcast.h"
#elif defined(BPMF_MPI_ISENDIRECV_COMM)
#define BPMF_MPI_COMM
#include "mpi_isendirecv.h"
#elif defined(BPMF_NO_COMM)
#include "nocomm.h"
#else
#error no comm include
#endif


void usage() 
{
    std::cout << "Usage: bpmf -n <MTX> -p <MTX> [-o DIR/] [-i N] [-b N] [-f N] [-krv] [-t N] [-m MTX,MTX] [-l MTX,MTX]\n"
                << "\n"
                << "Paramaters: \n"
                << "  -n MTX: Training input data\n"
                << "  -p MTX: Test input data\n"
                << "  [-o DIR]: Output directory for model and predictions\n"
                << "  [-i N]: Number of total iterations\n"
                << "  [-b N]: Number of burnin iterations\n"
                << "  [-f N]: Frequency to send model other nodes (in #iters)\n"
                << "  [-a F]: Noise precision (alpha)\n"
                << "\n"
                << "  [-k]: Do not optimize item to node assignment\n"
                << "  [-r]: Redirect stdout to file\n"
                << "  [-v]: Output all samples\n"
                << "  [-t N]: Number of OpenMP threads to use.\n"
                << "\n"
                << "  [-l MTX,MTX]: propagated posterior mu and Lambda matrices for U\n"
                << "  [-m MTX,MTX]: propagated posterior mu and Lambda matrices for V\n"
                << "\n"
                << "Matrix Formats:\n"
                << "  *.mtx: Sparse or dense Matrix Market format\n"
                << "  *.sdm: Sparse binary double format\n"
                << "  *.ddm: Dense binary double format\n"
                << std::endl;
}


int main(int argc, char *argv[])
{
    // If ARGO is not defined, initialize
    // the execution environment here
    //#ifndef BPMF_ARGO_COMM
        Sys::Init();
    //#endif

    {
        int ch;
        string fname, probename;
        string mname, lname;
        int nthrds = -1;
        bool redirect = false;
        Sys::nsims = 20;
        Sys::burnin = 5;
        Sys::update_freq = 1;
        Sys::grain_size = 1;

        while((ch = getopt(argc, argv, "krvn:t:p:i:b:f:g:w:u:v:o:s:m:l:a:d:")) != -1)
        {
            switch(ch)
            {
                case 'i': Sys::nsims = atoi(optarg); break;     // Number of total iterations
                case 'b': Sys::burnin = atoi(optarg); break;    // Number of burnin iterations
                case 'f': Sys::update_freq = atoi(optarg); break;
                case 'g': Sys::grain_size = atoi(optarg); break;
                case 't': nthrds = atoi(optarg); break;         // Number of OMP threads to use
                case 'a': Sys::alpha = atof(optarg); break;     // Noise precision (alpha)
                case 'd': assert(num_latent == atoi(optarg)); break;
                case 'n': fname = optarg; break;                // Train input data
                case 'p': probename = optarg; break;            // Test input data

                // Output directory matrices
                case 'o': Sys::odirname = optarg; break;        // Output directory for model and predictions

                case 'm': mname = optarg; break;
                case 'l': lname = optarg; break;

                case 'r': redirect = true; break;               // Redirect stdout to file
                case 'k': Sys::permute = false; break;
                case 'v': Sys::verbose = true; break;
                case '?':
                case 'h': 
                default : usage(); Sys::Abort(1);               // Print --help and abort
            }
        }

        // If ARGO is defined, initialize
        // the execution environment here
        // The below is done for argo::init
        /*
        #ifdef BPMF_ARGO_COMM
            unsigned long long init_num_latent;
            unsigned long long init_num_users;
            unsigned long long init_num_movies;

            {
                std::cout << "Reading dummy matrix..." << std::endl;

                SparseMatrixD M, T;     // We don't need these after we get the maxes
                read_matrix(fname, M);  // >>

                std::cout << "Finished dummy matrix..." << std::endl;

                init_num_latent = num_latent;
                init_num_users = std::max(M.rows(), T.rows());  // Taking into account train.mtx > test.mtx
                init_num_movies = std::max(M.cols(), T.cols()); // >>
            }

            Sys::Init(init_num_latent, init_num_users, init_num_movies);
        #endif
        */

        if (Sys::nprocs >1 || redirect) {
            std::stringstream ofname;
            ofname << "bpmf_" << Sys::procid << ".out";
            Sys::os = new std::ofstream(ofname.str());
        } else {
            Sys::os = &std::cout;
        }

        if (fname.empty() || probename.empty()) {
            usage();
            Sys::Abort(1);
        }

        auto begin_api = tick();

        // Reads fname into M and probename into T
        SYS movies("movs", fname, probename);
        // Transposes M and Pm2 = Pavg = T = Torig
        SYS users("users", movies.M, movies.Pavg);
        
        movies.add_prop_posterior(mname);
        users.add_prop_posterior(lname);

        movies.alloc_and_init();
        users.alloc_and_init();

        // Assign users and movies to the computation nodes
        movies.assign(users);
        users.assign(movies);
        movies.assign(users);
        users.assign(movies);

        /*
        #ifdef BPMF_ARGO_NO_COMM
            std::size_t movies_chunk = movies.num() / Sys::nprocs;
            std::size_t users_chunk = users.num() / Sys::nprocs;
            std::size_t data_begin;

            for (std::size_t i = 0; i < Sys::nprocs; ++i) {
                data_begin = i * movies_chunk;
                movies.dom[i] = data_begin;
                
                data_begin = i * users_chunk;
                users.dom[i] = data_begin;
            }
            
            movies.dom[Sys::nprocs] = movies.num();
            users.dom[Sys::nprocs] = users.num();
        #endif
        */

        //std::cout << "(MOV) Process " << Sys::procid << " from " << movies.from() << " to " << movies.to() << std::endl;
        //std::cout << "(USR) Process " << Sys::procid << " from " << users.from() << " to " << users.to() << std::endl;

        // Build connectivity matrix
        // Contains what items needs to go to what nodes
        users.build_conn(movies);
        movies.build_conn(users);
        assert(movies.nnz() == users.nnz());

        threads::init(nthrds);

        long double average_items_sec = .0;
        long double average_ratings_sec = .0;
    
        char name[1024];
        gethostname(name, 1024);
        Sys::cout() << "hostname: " << name << endl;
        Sys::cout() << "pid: " << getpid() << endl;
        if (getenv("PBS_JOBID")) Sys::cout() << "jobid: " << getenv("PBS_JOBID") << endl;

        if(Sys::procid == 0)
        {
            Sys::cout() << "num_latent: " << num_latent<<endl;
            Sys::cout() << "nprocs: " << Sys::nprocs << endl;
            Sys::cout() << "nthrds: " << threads::get_max_threads() << endl;
            Sys::cout() << "nsims: " << Sys::nsims << endl;
            Sys::cout() << "burnin: " << Sys::burnin << endl;
            Sys::cout() << "grain_size: " << Sys::grain_size << endl;
            Sys::cout() << "alpha: " << Sys::alpha << endl;
        }

        Sys::sync();

        auto begin = tick();

        for(int i=0; i<Sys::nsims; ++i) {
            BPMF_COUNTER("main");
            auto start = tick();

            {
                BPMF_COUNTER("movies");

                // Sample hyper-parameters movies based on V
                movies.sample_hp();
                // Update movie model m based on ratings (R) for this movie and model of users that rated this movie, plus randomly sampled noise
                movies.sample(users);
            }

            {
                BPMF_COUNTER("users");
                
                // Sample hyper-parameters users based on U
                users.sample_hp();
                // Update user u based on ratings (R) for this user and model of movies this user rated, plus randomly sampled noise
                users.sample(movies);
            }

            {
                BPMF_COUNTER("eval");
                
                // Predict rating and compute RMSE
                movies.predict(users);
                // Predict rating and compute RMSE
                users.predict(movies);
            }

            auto stop = tick();
            double items_per_sec = (users.num() + movies.num()) / (stop - start);
            double ratings_per_sec = (users.nnz()) / (stop - start);
            movies.print(items_per_sec, ratings_per_sec, sqrt(users.aggr_norm()), sqrt(movies.aggr_norm()));
            average_items_sec += items_per_sec;
            average_ratings_sec += ratings_per_sec;

            if (Sys::verbose)
            {
                #ifdef BPMF_ARGO_COMM
                    write_matrix(Sys::odirname + "/U-" + std::to_string(i) + ".ddm", users.items());
                    write_matrix(Sys::odirname + "/V-" + std::to_string(i) + ".ddm", movies.items());
                #else
                    users.bcast();
                    write_matrix(Sys::odirname + "/U-" + std::to_string(i) + ".ddm", users.items());
                    movies.bcast();
                    write_matrix(Sys::odirname + "/V-" + std::to_string(i) + ".ddm", movies.items());
                #endif
            }
        }

        Sys::sync();

        auto end = tick();
        auto elapsed = end - begin;

        // If we need to generate output files, collect all data on proc 0
        if (Sys::odirname.size()) {
            #ifdef BPMF_ARGO_COMM
                users.argo_bcast();
                movies.argo_bcast();
            #else
                users.bcast();
                movies.bcast();
            #endif

            movies.predict(users, true);

            // Restore original order
            users.unpermuteCols(movies);
            movies.unpermuteCols(users);

            if (Sys::procid == 0) {
                // Sparse
                write_matrix(Sys::odirname + "/Pavg.sdm", movies.Pavg);
                write_matrix(Sys::odirname + "/Pm2.sdm", movies.Pm2);

                #ifdef BPMF_ARGO_COMM
                    // Dense
                    users.finalize_mu_lambda();
                    write_matrix(Sys::odirname + "/U-mu.ddm", users.aggrMus());
                    write_matrix(Sys::odirname + "/U-Lambda.ddm", users.aggrLambdas());

                    movies.finalize_mu_lambda();
                    write_matrix(Sys::odirname + "/V-mu.ddm", movies.aggrMus());
                    write_matrix(Sys::odirname + "/V-Lambda.ddm", movies.aggrLambdas());
                #else
                    // Dense
                    users.finalize_mu_lambda();
                    write_matrix(Sys::odirname + "/U-mu.ddm", users.aggrMu);
                    write_matrix(Sys::odirname + "/U-Lambda.ddm", users.aggrLambda);

                    movies.finalize_mu_lambda();
                    write_matrix(Sys::odirname + "/V-mu.ddm", movies.aggrMu);
                    write_matrix(Sys::odirname + "/V-Lambda.ddm", movies.aggrLambda);
                #endif
            }
        }

        auto end_api = tick();
        auto elapsed_api = end_api - begin_api;
    
        if (Sys::procid == 0) {
            Sys::cout() << "Total API: " << elapsed_api <<endl <<flush;
            Sys::cout() << "Total BPMF: " << elapsed <<endl <<flush;
            Sys::cout() << "Final Avg RMSE: " << movies.rmse_avg <<endl <<flush;
            Sys::cout() << "Average items/sec: " << average_items_sec / movies.iter << endl <<flush;
            Sys::cout() << "Average ratings/sec: " << average_ratings_sec / movies.iter << endl <<flush;
        }
    }

    Sys::Finalize();
    if (Sys::nprocs >1) delete Sys::os;

    return 0;
}


void Sys::bcast()
{
    for(int i = 0; i < num(); i++) {
#ifdef BPMF_MPI_COMM
        MPI_Bcast(items().col(i).data(), num_latent, MPI_DOUBLE, proc(i), MPI_COMM_WORLD);
        if (aggrMu.nonZeros())
            MPI_Bcast(aggrMu.col(i).data(), num_latent, MPI_DOUBLE, proc(i), MPI_COMM_WORLD);
        if (aggrLambda.nonZeros())
            MPI_Bcast(aggrLambda.col(i).data(), num_latent*num_latent, MPI_DOUBLE, proc(i), MPI_COMM_WORLD);
#else
        assert(Sys::nprocs == 1);
#endif
    }
}


void Sys::finalize_mu_lambda()
{
    #ifdef BPMF_ARGO_COMM
        assert(aggrLambdas().nonZeros());
        assert(aggrMus().nonZeros());
        // Calculate real mu and Lambda
        for(int i = 0; i < num(); i++) {
            int nsamples = Sys::nsims - Sys::burnin;
            auto sum = aggrMus().col(i);
            auto prod = Eigen::Map<MatrixNNd>(aggrLambdas().col(i).data());
            MatrixNNd cov = (prod - (sum * sum.transpose() / nsamples)) / (nsamples - 1);
            MatrixNNd prec = cov.inverse(); // precision = covariance^-1
            aggrLambdas().col(i) = Eigen::Map<Eigen::VectorXd>(prec.data(), num_latent * num_latent);
            aggrMus().col(i) = sum / nsamples;
        }
    #else
        assert(aggrLambda.nonZeros());
        assert(aggrMu.nonZeros());
        // Calculate real mu and Lambda
        for(int i = 0; i < num(); i++) {
            int nsamples = Sys::nsims - Sys::burnin;
            auto sum = aggrMu.col(i);
            auto prod = Eigen::Map<MatrixNNd>(aggrLambda.col(i).data());
            MatrixNNd cov = (prod - (sum * sum.transpose() / nsamples)) / (nsamples - 1);
            MatrixNNd prec = cov.inverse(); // precision = covariance^-1
            aggrLambda.col(i) = Eigen::Map<Eigen::VectorXd>(prec.data(), num_latent * num_latent);
            aggrMu.col(i) = sum / nsamples;
        }
    #endif
}
