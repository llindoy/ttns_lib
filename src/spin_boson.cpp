#define TIMING

//#define USE_OLD
#define TTNS_REGISTER_COMPLEX_DOUBLE_OPERATOR

#ifdef CEREAL_LIBRARY_FOUND
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/binary.hpp>
#endif

#include "mode_data.hpp"
//#include "spectral_density.hpp"
//#include "spectral_density_orthopol_discretisation.hpp"

#include <ttns_lib/ttns.hpp>

#include <map>
#include <fstream>
#include <sstream>
#include <random>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <algorithm>
#include <ttns_lib/tdvp_core/two_site_energy_variations.hpp>


#include "bath/bath_types/exponential.hpp"
#include "bath/bath_types/sudden.hpp"
#include "bath/bath_types/gaussian.hpp"
#include "bath/bath_types/debye.hpp"
#include "bath/bath_types/jacobi_cutoff.hpp"

#include <orthopol.hpp>
#include <io/input_wrapper.hpp>

using namespace ttns;


template <typename integ_type, typename complex_type, typename real_type, typename backend_type, typename op_type>
void run(httensor<complex_type, backend_type>& A, integ_type& tdvp, sop_operator<complex_type, backend_type>& H, std::vector<op_type>& mops, std::vector<linalg::vector<real_type>>& SzSz, real_type tmax, std::ostream& os, bool print_first = false, bool print_hrank = false)
{
    os << std::setprecision(16);
    if(print_first)
    {
        auto& mel = tdvp.mel();
        os << 0.0 << " ";
        for(size_t j=0; j < mops.size(); ++j)
        {
            SzSz[j](0) += linalg::real(mel(mops[j], 0, A));
            os << SzSz[j](0) << " ";
        }
        os <<  " 0 0 0 ";
        if(print_hrank)
        {
            for(auto& c : A)
            {
                if(!c.is_root())
                {
                    os << c.hrank() << " " ;
                }
            }
        }
        os << std::endl;    
    }
    linalg::matrix<complex_type, backend_type> At;
    for(auto& c : A)
    {
        if(c.is_leaf())
        {
            if(c.leaf_index() == 0)
            {
                At.resize(c().shape(0), c().shape(1));
            }
        }
    }
    A.set_is_orthogonalised();
    size_t ncount = 1;
    //variable time step integrator
    while(tdvp.t() < tmax)
    {
        tdvp(A, H, complex_type(0, -1));
        auto& mel = tdvp.mel();
        os << tdvp.t() << " ";
        for(size_t j=0; j < mops.size(); ++j)
        {
            SzSz[j](ncount) += linalg::real(mel(mops[j], 0, A));
            os << SzSz[j](ncount) << " ";
        }
        os << tdvp.engine().Nonesite() << " " << tdvp.engine().Ntwosite() << " ";
        os << tdvp.nh_applications()/static_cast<real_type>(A.size()) << " ";
        if(print_hrank)
        {
            for(auto& c : A)
            {
                if(!c.is_root())
                {
                    os << c.hrank() << " " ;
                }
            }
        }
        os << std::endl;
        ++ncount;
    }
    os << "nh_app: " << tdvp.nh_applications()/static_cast<real_type>(A.size()) << std::endl;
}


int main(int argc, char* argv[])
{
    try
    {
        using real_type = double;
        using complex_type = ttns::complex<real_type>;
        using backend_type = linalg::blas_backend;
        using size_type = typename backend_type::size_type;
        backend_type::initialise();

        if(argc < 2)
        {
            std::cerr << argv[0] << " <input filename>" << std::endl;
            std::cerr << io::factory<bath::continuous_bath<real_type>>::get_all_info() << std::endl;
            return 1;
        }

        std::ifstream ifs(argv[1]);
        if(!ifs.is_open())
        {
            std::cerr << "Could not open input file." << std::endl;
            return 1;
        }
        using IObj = IOWRAPPER::input_base;

        IObj doc {};
        IOWRAPPER::parse_stream(doc, ifs);

        //read in the inputs
        std::shared_ptr<bath::continuous_bath<real_type>> exp;

        real_type tmax, dt;
        CALL_AND_HANDLE(IOWRAPPER::load<real_type>(doc, "tmax", tmax), "Failed to load maximum integration time.");
        CALL_AND_HANDLE(IOWRAPPER::load<real_type>(doc, "dt", dt), "Failed to load integration timestep.");

        size_type nspf, nspf_lower, nmax_dim, ntarget, N;
        CALL_AND_HANDLE(IOWRAPPER::load<size_type>(doc, "nspf", nspf), "Failed to load the number of single particle functions for the top level bath nodes.");
        CALL_AND_HANDLE(IOWRAPPER::load<size_type>(doc, "nspflower", nspf_lower), "Failed to load the number of single particle functions for hte bottom level bath nodes.");
        CALL_AND_HANDLE(IOWRAPPER::load<size_type>(doc, "maximumdimension", nmax_dim), "Failed to load the maximum local hilbert space dimension for a mode.");
        CALL_AND_HANDLE(IOWRAPPER::load<size_type>(doc, "targetdimension", ntarget), "Failed to load the target hilbert space dimension.");
        CALL_AND_HANDLE(IOWRAPPER::load<size_type>(doc, "nmodes", N), "Failed to load the number of bath modes used for discretisation.");

        ASSERT(IOWRAPPER::has_member(doc, "bath"), "Spectral density not found.");
        CALL_AND_HANDLE(exp =  io::factory<bath::continuous_bath<real_type>>::create(doc["bath"]), "Failed to read in bath spectral density.");
        

        size_type seed = 0;
        CALL_AND_HANDLE(IOWRAPPER::load_optional<size_type>(doc, "seed", seed), "Failed to load the random number generator seed.");

        size_type nmaxlargefreq = nmax_dim;
        CALL_AND_HANDLE(IOWRAPPER::load_optional<size_type>(doc, "maximumdimensionhighfrequency", nmaxlargefreq), "Failed to load the maximum local hilbert space dimension for high frequency modes.");

        real_type wc = 10.0;
        CALL_AND_HANDLE(IOWRAPPER::load_optional<real_type>(doc, "highfrequencybound", wc), "Failed to load the frequency to change over from low to high frequency.");


        bool user_specified_cutoff = false;
        real_type maximum_included_frequency = 0;
        CALL_AND_HANDLE(user_specified_cutoff = IOWRAPPER::load_optional<real_type>(doc, "maxfrequencycutoff", maximum_included_frequency), "Failed to load the hard frequency cutoff.");


        bool user_specified_minimum_cutoff = false;
        real_type negative_included_frequency = 0;
        CALL_AND_HANDLE(user_specified_minimum_cutoff = IOWRAPPER::load_optional<real_type>(doc, "negativefrequencycutoff", negative_included_frequency), "Failed to load the hard frequency cutoff.");

        real_type btol = 1e-5;
        CALL_AND_HANDLE(IOWRAPPER::load_optional<real_type>(doc, "bathintegrationerrortolerance", btol), "Failed to load the random number generator seed.");

        //now read in the system Hamiltonian
        linalg::matrix<complex_type, backend_type> Hsys;
        CALL_AND_HANDLE(IOWRAPPER::load<decltype(Hsys)>(doc, "hsys", Hsys), "Failed to load the system hamiltonian.");
        ASSERT(Hsys.shape(0) == Hsys.shape(1), "The system hamiltonian is not square.");
        size_type nhilb = Hsys.shape(0);


        linalg::vector<complex_type, backend_type> psi0;
        CALL_AND_HANDLE(IOWRAPPER::load<decltype(psi0)>(doc, "psi0", psi0), "Failed to load the system wavefunction.");
        ASSERT(psi0.shape(0) == nhilb, "The system wavefunction is not the correct size.");

    
        //now read in the system bath coupling matrix
        linalg::matrix<complex_type, backend_type> Scoup;
        CALL_AND_HANDLE(IOWRAPPER::load<decltype(Scoup)>(doc, "scoup", Scoup), "Failed to load the system contribution to the system-bath coupling hamiltonian.");
        ASSERT(Scoup.shape(0) == nhilb && Scoup.shape(1) == nhilb, "The system contribution to the system-bath coupling operator must be a square matrix.");

        real_type ecut = 3000;

        size_type nthreads = 1;
        CALL_AND_HANDLE(IOWRAPPER::load_optional<size_type>(doc, "nthreads", nthreads), "Failed to load nthreads.");
        blas_set_num_threads(nthreads);
        omp_set_num_threads(nthreads);


        real_type moment_scaling = 2;
        CALL_AND_HANDLE(IOWRAPPER::load_optional<real_type>(doc, "momentscaling", moment_scaling), "Failed to load krylov tolerance.");

        real_type krylov_tolerance = 1e-12;
        CALL_AND_HANDLE(IOWRAPPER::load_optional<real_type>(doc, "krylovtolerance", krylov_tolerance), "Failed to load krylov tolerance.");

        size_type krylov_dim = 6;
        CALL_AND_HANDLE(IOWRAPPER::load_optional<size_type>(doc, "krylovdimension", krylov_dim), "Failed to load krylov dimension.");

        real_type spawning_parameter = 1e-12;
        CALL_AND_HANDLE(IOWRAPPER::load_optional<real_type>(doc, "spawningparameter", spawning_parameter), "Failed to load spawning parameter.");

        real_type unoccupied_threshold = 1e-12;
        CALL_AND_HANDLE(IOWRAPPER::load_optional<real_type>(doc, "unoccupiedthreshold", unoccupied_threshold), "Failed to load unoccupied threshold parameter.");

        real_type jalpha = 1.0;
        CALL_AND_HANDLE(IOWRAPPER::load_optional<real_type>(doc, "jacobialpha", jalpha), "Failed to load unoccupied threshold parameter.");
        real_type jbeta = 0.0;
        CALL_AND_HANDLE(IOWRAPPER::load_optional<real_type>(doc, "jacobibeta", jbeta), "Failed to load unoccupied threshold parameter.");

        size_type nspf_cap = nspf;
        CALL_AND_HANDLE(IOWRAPPER::load_optional<size_type>(doc, "nspfcapacity", nspf_cap), "Failed to load nspf capacity.");
        if(nspf_cap < nspf){nspf_cap = nspf;}
        
        size_type nspf_lower_cap = nspf_lower;
        CALL_AND_HANDLE(IOWRAPPER::load_optional<size_type>(doc, "nspflowercapacity", nspf_lower_cap), "Failed to load nspf lower capacity.");
        if(nspf_lower_cap < nspf_lower){nspf_lower_cap = nspf_lower;}
        
        size_type minimum_unoccupied = nspf_cap;
        CALL_AND_HANDLE(IOWRAPPER::load_optional<size_type>(doc, "minimumunoccupied", minimum_unoccupied), "Failed to load the minimum number of unoccupied elements.");

        bool print_hrank = false;
        CALL_AND_HANDLE(IOWRAPPER::load_optional<bool>(doc, "printbonddimension", print_hrank ), "Failed to load whether or not to print the bond dimension of tensors..");

        bool has_ofile = false;
        std::string ofilename;
        CALL_AND_HANDLE(has_ofile = IOWRAPPER::load_optional<std::string>(doc, "outputfile", ofilename), "Failed to load the minimum number of unoccupied elements.");

        std::ofstream ofs;
        if(has_ofile)
        {
            ofs.open(ofilename.c_str());
        }

        size_type Npoly = N;

        //in general we will need to scale the 
        orthopol<real_type> n_poly;   

        bath::continuous_bath<double>::fourier_integ_type integ(10, 100);
        std::array<real_type, 2> bounds ;
        if(user_specified_cutoff) 
        {
            bounds[1] = maximum_included_frequency;
            bounds[0] = -bounds[1];
        }
        else
        {
            bounds = exp->frequency_bounds(btol, integ.quad(), true);
        }

        //need to figure out how to do the rescaling correctly.  The quadrature rules we are generating are incorrect
        if(exp->nonzero_temperature())
        {
            orthopol<real_type> cheb;     jacobi_polynomial(cheb, 2*Npoly, jalpha, jbeta);//, 1.0, 0.0);

            if(user_specified_minimum_cutoff)
            {
                bounds[0] = negative_included_frequency;
            }
    
            real_type wmax = bounds[1]; real_type wmin = bounds[0];
            real_type wrange = wmax - wmin;
            //now shift the chebyshev functions
            cheb.shift((wmax+wmin)/wrange);
            cheb.scale(moment_scaling);
            nonclassical_polynomial(n_poly, cheb, Npoly, [exp, &wrange, &moment_scaling](real_type x){return 0.5*(exp->S(x*wrange/(2*moment_scaling)) + exp->J(x*wrange/(2*moment_scaling)));}, 1e-9);
            n_poly.scale(wrange/(2*moment_scaling));

        }
        else
        {
            orthopol<real_type> cheb;     jacobi_polynomial(cheb, 2*Npoly, jalpha, jbeta);//, 1.0, 0.0);
            real_type wmax = bounds[1];    real_type wmin = 0;
            real_type wrange = wmax - wmin;

            cheb.shift((wmax+wmin)/wrange);
            cheb.scale(moment_scaling);
            nonclassical_polynomial(n_poly, cheb, Npoly, [exp, &wrange, &moment_scaling](real_type x){return exp->J(x*wrange/(2*moment_scaling));}, 1e-10);
            n_poly.scale(wrange/(2.0*moment_scaling));
        }

        n_poly.compute_nodes_and_weights();

        //setup the bath parameters
        //the spin boson model parameters
        
        std::vector<std::pair<real_type, real_type>> wg(N);
        for(size_type i = 0; i < N; ++i)
        {
            wg[i] = std::make_pair(n_poly.nodes()(i), n_poly.weights()(i));
        }

        std::sort(wg.begin(), wg.end(), [](const std::pair<real_type, real_type>& a, const std::pair<real_type, real_type>& b){return std::abs(std::get<0>(a)) < std::abs(std::get<0>(b));});

        std::vector<real_type> _wk(N);   std::vector<real_type> _gk(N);
        std::cerr << "w_k \t g_k" << std::endl;
        for(size_type i = 0; i < N; ++i)
        {
            _wk[i] = std::get<0>(wg[i]);
            _gk[i] = std::sqrt(std::get<1>(wg[i])/M_PI);    

            std::cerr << _wk[i] << " " << _gk[i] << std::endl;
        }
        
        N = _wk.size();
        std::vector<real_type> wk(N);

        for(size_t i = 0; i < N; ++i)
        {
            for(size_t j=0; j < 1; ++j)
            {
                wk[i*1+j] = _wk[i];
            }
        }

        std::vector<std::vector<real_type>> gk(1);
        for(size_t i = 0; i < 1; ++i)
        {
            gk[i].resize(N*1);
            for(size_t j = 0; j < N; ++j)
            {
                for(size_t k=0; k < 1; ++k)
                {
                    gk[i][j*1+k] = _gk[j];
                }
            }
        }
        std::cerr << "bath discretised." << std::endl;

        //start constructing the tree topology.  
        size_type nbranch = 2;


        //determine the partitioning of the N modes into nblocks so that they all have roughly the same numbers of states.
        std::vector<size_t> partitions;
        partition_modes(wk, ecut, 2*wc, partitions, ntarget, 250, nmax_dim, nmaxlargefreq);
        size_t nblocks = partitions.size();
        std::vector<size_t> nmodes(nblocks);
        std::vector<size_t> mode_dimensions(nblocks+1);
        harmonic_mode_combination<real_type> mode_combination;
        std::cout << nblocks << std::endl;

        mode_dimensions[0] = nhilb;

        size_t counter;
        std::vector<size_t> nskip(nblocks);
        {
            counter =0;
            for(size_t i=0; i<nblocks; ++i)
            {
                nskip[i] = counter;
                size_t nmaxt = wk[nskip[i]] > 2*wc ? nmaxlargefreq : nmax_dim;
                mode_combination.set_wk(wk.begin()+counter, wk.begin()+counter + partitions[i]); 
                mode_combination.set_gk(gk[0].begin()+counter, gk[0].begin()+counter + partitions[i]); 
                size_t nstates = mode_combination.construct_basis_topology(ecut, nmaxt);

                nmodes[i] = nstates;
                mode_dimensions[i+1] = nstates;
                counter += partitions[i];
            }
        }

        //now we build the topology tree for 
        ntree<size_type> topology{};    topology.insert(1);
        topology().insert(mode_dimensions[0]);        topology()[0].insert(mode_dimensions[0]);
        topology().insert(mode_dimensions[0]);

        size_type nlevels = static_cast<size_type>(std::log2(nmodes.size()));
        ntree_builder<size_type>::htucker_subtree(topology()[1], nmodes, nbranch, 
        [nspf, nspf_lower, nlevels](size_type l)
        {
            size_type ret = 0;
            if( l >= nlevels){ret = nspf_lower;}
            else if(l == 0){ret = nspf;}
            else
            {
                real_type rmax = std::log2(nspf);
                real_type rmin = std::log2(nspf_lower);
                ret = static_cast<size_type>(std::pow(2.0, ((nlevels-l)*static_cast<real_type>(rmax-rmin))/nlevels+rmin));
            }
            return ret;
        }
        );
        ntree_builder<size_type>::sanitise_tree(topology);

        ntree<size_type> capacity{};    capacity.insert(1);
        capacity().insert(mode_dimensions[0]);        capacity()[0].insert(mode_dimensions[0]);
        capacity().insert(mode_dimensions[0]);

        ntree_builder<size_type>::htucker_subtree(capacity()[1], nmodes, nbranch, 
        [nspf_cap, nspf_lower_cap, nlevels](size_type l)
        {
            size_type ret = 0;
            if( l >= nlevels){ret = nspf_lower_cap;}
            else if(l == 0){ret = nspf_cap;}
            else
            {
                real_type rmax = std::log2(nspf_cap);
                real_type rmin = std::log2(nspf_lower_cap);
                ret = static_cast<size_type>(std::pow(2.0, ((nlevels-l)*static_cast<real_type>(rmax-rmin))/nlevels+rmin));
            }
            return ret;
        }
        );
        ntree_builder<size_type>::sanitise_tree(capacity);
        std::cerr << "tree topology constructed" << std::endl;

        
        std::cerr << std::setprecision(16) << std::endl;
        //now we can construct our initial ttns representation of the wavefunction
        httensor<complex_type, backend_type> A(topology, capacity);     
        std::cerr << "htucker tensor built" << std::endl;

        //create the sum of product Hamiltonian
        std::vector<size_type> terms_per_mode(nblocks+1);
        for(auto& z : terms_per_mode){z = 1+1;}
        sop_operator<complex_type, backend_type> H( 1 + (1+1)*nblocks, mode_dimensions, terms_per_mode);

        H.bind(ops::dense_matrix_operator<complex_type, backend_type>{Hsys}, {0}, 0); //bind the system Hamiltonian

        std::vector<size_type> sigma_z_indices(nblocks);
        //bind the different spin coupling operators
        for(size_type spi = 0; spi < 1; ++spi)
        {
            for(size_type i=0; i  < nblocks; ++i)
            {
                sigma_z_indices[i] = 1 + (1+1)*i + spi + 1;
            }
            H.bind(ops::dense_matrix_operator<complex_type, backend_type>{Scoup}, sigma_z_indices, 0); 
        }

        for(size_type i=0; i<nblocks; ++i)
        {   
            size_t nmaxt = wk[nskip[i]] > 2*wc ? nmaxlargefreq : nmax_dim;
            //set up the required mode information
            mode_combination.set_wk(wk.begin()+nskip[i], wk.begin()+nskip[i] + partitions[i]); 
            mode_combination.set_gk(gk[0].begin()+nskip[i], gk[0].begin()+nskip[i] + partitions[i]); 
            mode_combination.construct_basis_topology(ecut, nmaxt);

            //now we set up the diagonal boson operator terms
            {
                linalg::diagonal_matrix<complex_type, backend_type> H0;
                //diagonal matrix
                mode_combination.H0(H0);
                H.bind(ops::diagonal_matrix_operator<complex_type, backend_type>(H0), {1 + (1+1)*i}, i+1);
            }

            //now we set up the coupling operator terms
            {
                //csr matrix
                linalg::csr_matrix<complex_type, backend_type> Hc;
                for(size_type spi = 0; spi < 1; ++spi)
                {
                    mode_combination.set_gk(gk[spi].begin()+nskip[i], gk[spi].begin()+nskip[i] + partitions[i]); 
                    mode_combination.Hc(Hc);
                    H.bind(ops::sparse_matrix_operator<complex_type, backend_type>(Hc),  {1 + (1+1)*i + spi + 1 }, i+1);
                }
            }
        }

        std::cerr << "sum of product operator initialised." << std::endl;

        std::vector<linalg::matrix<complex_type>> m_ops;
        ASSERT(IOWRAPPER::has_member(doc, "ops"), "Unable to find observable operators.");
        if(IOWRAPPER::is_type<linalg::matrix<complex_type>>(doc, "ops"))
        {
            m_ops.resize(1);
            CALL_AND_HANDLE(IOWRAPPER::load<linalg::matrix<complex_type>>(doc, "ops", m_ops[0]), "Failed to read in the observable operators.");
        }
        else if(IOWRAPPER::is_type<std::vector<linalg::matrix<complex_type>>>(doc, "ops"))
        {
            CALL_AND_HANDLE(IOWRAPPER::load<decltype(m_ops)>(doc, "ops", m_ops), "Failed to read in the observable operators.");
        }
        else
        {
            RAISE_EXCEPTION("Invalid observable operator type.");
        }

        std::vector<ops::dense_matrix_operator<complex_type, backend_type>> mops(m_ops.size());
        for(size_t i = 0; i < m_ops.size(); ++i)
        {
            ASSERT(m_ops[i].size(0) == nhilb && m_ops[i].size(1) == nhilb, "The size of the observable operator matrix is not the same as the system Hilbert space dimension.");
            mops[i] = ops::dense_matrix_operator<complex_type, backend_type>(m_ops[i]);
        }
        

      

        size_t nsteps = static_cast<size_t>(tmax/dt)+50;
    
        std::vector<linalg::vector<real_type>> SzSz(m_ops.size());
        for(size_type i = 0; i < m_ops.size(); ++i){SzSz[i].resize(nsteps);  SzSz[i].fill_zeros();}
        
        std::mt19937 rng(seed);

        ttns::two_site_variations<complex_type, backend_type> twosite;
        std::vector<size_t> state_prev(nblocks, 0);
        //evaluate the Sz1Szk correlation functions.  We need to sample over identity for the Sz1 state
        //initialise the wavefunction in the ground state of the bosonic system and in the |0> state of the electronic system
        for(auto& c : A)
        {
            if(c.is_leaf())
            {
                linalg::matrix<complex_type, backend_type> ct(c().shape(1), c().shape(0));  ct.fill_zeros();
                if(c.leaf_index() == 0)
                {
                    CALL_AND_HANDLE(ct[0] = psi0, "Failed to assign psi0 in htucker.");
                    //for(size_type i=0; i<c().size(0); ++i){){c()(i, j) = (i == j ? 1.0 : 0.0);}}
                    //for(size_type i=0; i<c().size(0); ++i){for(size_type j=0; j<c().size(1); ++j){c()(i, j) = (i == j ? 1.0 : 0.0);}}
                }
                else
                {
                    ct(0, 0) = 1.0;
                }

                std::uniform_real_distribution<real_type> dist(0, 2.0*acos(real_type(-1.0)));
                std::normal_distribution<real_type> length_dist(0, 1);
                for(size_type i=1; i<c().shape(1); ++i)
                {
                    bool vector_generated = false;
            
                    while(!vector_generated)
                    {
                        //generate a random vector
                        for(size_type j=0; j<c().shape(0); ++j)
                        {
                            real_type theta = dist(rng);
                            ct(i, j) = length_dist(rng)*complex_type(cos(theta), sin(theta));
                        }

                        //now we normalise it
                        ct[i] /= sqrt(dot_product(conj(ct[i]), ct[i]));

                        //now we attempt to modified gram-schmidt this
                        //if we run into linear dependence then we need to try another random vector
                        for(size_type j=0; j < i; ++j)
                        {
                            ct[i] -= dot_product(conj(ct[j]), ct[i])*ct[j];
                        }

                        //now we compute the norm of the new vector
                        real_type norm = sqrt(real(dot_product(conj(ct[i]), ct[i])));
                        if(norm > 1e-12)
                        {
                            ct[i] /= norm;
                            vector_generated = true;
                        }
                    }
                }
                c().as_matrix() = trans(ct);
                
            }
            //if its an interior node fill it with the identity matrix
            else if(!c.is_root())
            {
                for(size_type i=0; i<c().size(0); ++i){for(size_type j=0; j<c().size(1); ++j){c()(i, j) = (i == j ? 1.0 : 0.0);}}
            }
            //and fill the root node with the matrix with 1 at position 0, 0 and 0 everywhere else
            else
            {
                for(size_type i=0; i<c().size(0); ++i){for(size_type j=0; j<c().size(1); ++j){c()(i, j) = ((i==j)&&(i==0) ? 1.0 : 0.0);}}
            }
        }

        A.set_is_orthogonalised();

        //now we set up the tdvp integrator object
        subspace_expansion_projector_splitting_integrator<complex_type, backend_type> tdvp(A, H, krylov_dim, krylov_tolerance, nthreads);
        //auto& mel = tdvp.mel();
        
        tdvp.dt() = dt;
        tdvp.spawning_threshold() = spawning_parameter;
        tdvp.unoccupied_threshold() = unoccupied_threshold;
        tdvp.minimum_unoccupied() = minimum_unoccupied;

        

        if(has_ofile)
        {
            run(A, tdvp, H, mops, SzSz, tmax, ofs, true, print_hrank);
        }
        else
        {
            run(A, tdvp, H, mops, SzSz, tmax, std::cout, true, print_hrank);
        }
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        return 1;
    }

}




