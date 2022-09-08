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
#include <orthopol.hpp>

using namespace ttns;


template <typename integ_type, typename complex_type, typename real_type, typename backend_type, typename op_type>
void run(httensor<complex_type, backend_type>& A, integ_type& tdvp, sop_operator<complex_type, backend_type>& H, std::vector<op_type>& mops, std::vector<linalg::vector<real_type>>& SzSz, real_type tmax, std::ostream& os, bool print_first = false)
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
        for(auto& c : A)
        {
            if(!c.is_root())
            {
                os << c.hrank() << " " ;
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
        for(auto& c : A)
        {
            if(!c.is_root())
            {
                os << c.hrank() << " " ;
            }
        }
        os << std::endl;
        ++ncount;
    }
    os << "nh_app: " << tdvp.nh_applications()/static_cast<real_type>(A.size()) << std::endl;
}


void check_inputs(const rapidjson::Value& doc)
{
    ASSERT(doc.HasMember("tmax"), "tmax not found");
    ASSERT(doc.HasMember("dt"), "dt not found"); 
    ASSERT(doc.HasMember("nspf"), "nspf not found"); 
    ASSERT(doc.HasMember("nspflower"), "nspflower not found"); 
    ASSERT(doc.HasMember("maximumdimension"), "maximumdimension not found"); 
    ASSERT(doc.HasMember("targetdimension"), "targetdimension not found"); 
    ASSERT(doc.HasMember("n"), "n not found"); 
    ASSERT(doc.HasMember("spectraldensity"), "spectraldensity not found"); 
    ASSERT(doc.HasMember("maximumdimensionhighfreq"), "maximumdimensionhighfreq not found"); 
    ASSERT(doc.HasMember("seed"), "seed not found"); 
    ASSERT(doc.HasMember("eps"), "eps not found"); 
    ASSERT(doc.HasMember("delta"), "delta not found"); 
    ASSERT(doc.HasMember("btol"), "btol not found"); 
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
            return 1;
        }

        std::ifstream ifs(argv[1]);
        if(!ifs.is_open())
        {
            std::cerr << "Could not open input file." << std::endl;
            return 1;
        }

        rapidjson::IStreamWrapper isw{ifs};

        rapidjson::Document doc {};
        doc.ParseStream(isw);

        if(doc.HasParseError())
        {
            std::cerr << "Error: " << doc.GetParseError() << std::endl << "Offset: " << doc.GetErrorOffset() << std::endl;
            return 1;
        }
        
        eos::dfs_replace_space_and_capitals_in_key(doc, doc.GetAllocator());
        CALL_AND_HANDLE(check_inputs(doc), "Invalid input file.");

        //read in the inputs
        
        //read in the bath
        std::shared_ptr<eos::bath::continuous_bath<real_type>> exp;

        real_type tmax, dt, eps, btol, delta;
        size_type nspf, nspf_lower, nmax3, nmaxlargefreq, nmax, N, seed;
        real_type ecut = 3000;
        size_type axis_index = 2;
        real_type s = 1.0;
        size_type nspins = 1;
        real_type wc = 10;
        try
        {
            tmax = eos::rapidjson_loader<real_type>::load(doc["tmax"]);
            dt = eos::rapidjson_loader<real_type>::load(doc["dt"]);
            eps = eos::rapidjson_loader<real_type>::load(doc["eps"]);
            delta = eos::rapidjson_loader<real_type>::load(doc["delta"]);
            btol = eos::rapidjson_loader<real_type>::load(doc["btol"]);

            nspf = eos::rapidjson_loader<size_type>::load(doc["nspf"]);
            nspf_lower = eos::rapidjson_loader<size_type>::load(doc["nspflower"]);
            nmax3 = eos::rapidjson_loader<size_type>::load(doc["maximumdimension"]);
            nmax = eos::rapidjson_loader<size_type>::load(doc["targetdimension"]);
            N = eos::rapidjson_loader<size_type>::load(doc["n"]);
            nmaxlargefreq = eos::rapidjson_loader<size_type>::load(doc["maximumdimensionhighfreq"]);
            seed = eos::rapidjson_loader<size_type>::load(doc["seed"]);

            ASSERT(doc["spectraldensity"].IsObject(), "spectral density is invalid.");
            exp =  eos::factory<eos::bath::continuous_bath<real_type>>::create(doc["spectraldensity"]);
        }
        catch(const std::exception& ex)
        {
            std::cerr << "Failed to read input parameters from file." << std::endl;
            return 1;
        }

        bool has_axis = false;


        size_type nthreads = 1;

        real_type krylov_tolerance = 1e-12;
        {
            if(doc.HasMember("krylovtolerance"))
            {
                krylov_tolerance = eos::rapidjson_loader<real_type>::load(doc["krylovtolerance"]);
            }
        }

        real_type spawning_parameter = 1e-12;
        {
            if(doc.HasMember("spawningparameter"))
            {
                spawning_parameter = eos::rapidjson_loader<real_type>::load(doc["spawningparameter"]);
            }
        }
        real_type unoccupied_threshold = 1e-12;
        {
            if(doc.HasMember("unoccupiedthreshold"))
            {
                unoccupied_threshold = eos::rapidjson_loader<real_type>::load(doc["unoccupiedthreshold"]);
            }
        }

        size_type nspf_cap = nspf;
        {
            if(doc.HasMember("nspfcapacity"))
            {
                nspf_cap = eos::rapidjson_loader<real_type>::load(doc["nspfcapacity"]);
            }
            if(nspf_cap < nspf){nspf_cap = nspf;}
        }
        size_type nspf_lower_cap = nspf_lower;
        {
            if(doc.HasMember("nspflowercapacity"))
            {
                nspf_lower_cap = eos::rapidjson_loader<real_type>::load(doc["nspflowercapacity"]);
            }
            if(nspf_lower_cap < nspf_lower){nspf_lower_cap = nspf_lower;}
        }

        size_type minimum_unoccupied = nspf_cap;
        {
            if(doc.HasMember("minimumunoccupied"))
            {
                minimum_unoccupied = eos::rapidjson_loader<real_type>::load(doc["minimumunoccupied"]);
            }
        }

        //set up the system operators
        size_t nhilb = nspins+1;


        //maybe set these up as sparse matrices instead
        //here we actually set this up so that we form 2*S_\alpha
        linalg::matrix<complex_type, backend_type> Sx(nhilb, nhilb);
        linalg::matrix<complex_type, backend_type> Sy(nhilb, nhilb);
        linalg::matrix<complex_type, backend_type> Sz(nhilb, nhilb);

        linalg::matrix<complex_type, backend_type> Hcoup(nhilb, nhilb);
        linalg::matrix<complex_type, backend_type> Hcoup2(nhilb, nhilb);
        linalg::matrix<complex_type, backend_type> Stot(nhilb, nhilb);
    
        //set up each of the spin 
        {
            linalg::matrix<complex_type> hSxi(nhilb, nhilb);
            linalg::matrix<complex_type> hSyi(nhilb, nhilb);
            linalg::matrix<complex_type> hSzi(nhilb, nhilb);
            linalg::matrix<complex_type> hStot(nhilb, nhilb);

            hSxi.fill_zeros();
            hSyi.fill_zeros();
            hSzi.fill_zeros();
            hStot.fill_zeros();
            real_type S2 = nspins/2.0*(nspins/2.0+1.0); 
            for(size_t j=0; j < nhilb; ++j)
            {
                hStot(j, j) = S2;

                real_type m = nspins/2.0-j;
                if(j+1 != nhilb)
                {
                    hSxi(j, j+1) = sqrt(S2-m*(m-1));
                    hSxi(j+1, j) = sqrt(S2-m*(m-1));

                    hSyi(j, j+1) = sqrt(S2-m*(m-1))/complex_type(0, 1);
                    hSyi(j+1, j) = sqrt(S2-m*(m-1))/complex_type(0, -1);
                }

                hSzi(j, j) = 2.0*(nspins/2.0-j);
            }   
            Sx = hSxi;
            Sy = hSyi;
            Sz = hSzi;
            Stot = hStot;
        }


        size_type Npoly = N;
        //orthopol<real_type> poly;     logx_weight_polynomial(poly, Npoly);//, 1.0, 0.0);
        //poly.scale(0.5);


        //in general we will need to scale the 
        orthopol<real_type> n_poly;   

        eos::bath::continuous_bath<double>::fourier_integ_type integ(10, 100);
        std::array<real_type, 2> bounds = exp->frequency_bounds(btol, integ.quad(), true);
        real_type wmax = bounds[1]; real_type wmin = bounds[0];
        std::cerr << wmin << " " << wmax << std::endl;
        //need to figure out how to do the rescaling correctly.  The quadrature rules we are generating are incorrect
        if(exp->nonzero_temperature())
        {
            orthopol<real_type> cheb;     jacobi_polynomial(cheb, 2*Npoly, 0.0, 0.0);//, 1.0, 0.0);

            real_type wrange = wmax - wmin;
            std::cerr << wmin << " " <<  wmax << " " << wrange << std::endl;
            //now shift the chebyshev functions
            cheb.shift((wmax+wmin)/wrange);
            cheb.scale(2.0);
            nonclassical_polynomial(n_poly, cheb, Npoly, [exp, &wrange](real_type x){return 0.5*(exp->S(x*wrange/4) + exp->J(x*wrange/4));}, 1e-9);
            n_poly.scale(wrange/4.0);

        }
        else
        {
            orthopol<real_type> cheb;     jacobi_polynomial(cheb, 2*Npoly, 1.0, 0.0);//, 1.0, 0.0);

            real_type wmin = 0;
            real_type wrange = wmax - wmin;

            cheb.shift((wmax+wmin)/wrange);
            cheb.scale(2.0);
            nonclassical_polynomial(n_poly, cheb, Npoly, [exp, &wrange](real_type x){return exp->J(x*wrange/4);}, 1e-10);
            n_poly.scale(wrange/4.0);
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
        real_type sumgk = 0.0;
        for(size_type i = 0; i < N; ++i)
        {
            _wk[i] = std::get<0>(wg[i]);
            _gk[i] = std::sqrt(std::get<1>(wg[i])/M_PI);    
            sumgk += _gk[i];
            std::cout << _wk[i] << " " << _gk[i] << std::endl;
        }
        exit(1);

        //std::cout << "integral " << sumgk << std::endl;

        linalg::matrix<complex_type, backend_type> Hsys(nhilb, nhilb);  Hsys.fill_zeros();

        Hcoup = Sz;
        Hsys = Sz*eps+Sx*delta;
        
    
        real_type max_coupling = 0;
        for(size_t i = 0; i < _wk.size(); ++i)
        {
            if(std::abs(_gk[i]) > max_coupling)
            {
                max_coupling = std::abs(_gk[i]);
            }
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

        //start constructing the tree topology.  
        size_type nbranch = 2;


        //determine the partitioning of the N modes into nblocks so that they all have roughly the same numbers of states.
        std::vector<size_type> partitions;
        partition_modes(wk, ecut, 2*wc, partitions, nmax, 250, nmax3, nmaxlargefreq);
        size_t nblocks = partitions.size();
        std::vector<size_type> nmodes(nblocks);
        std::vector<size_type> mode_dimensions(nblocks+1);
        harmonic_mode_combination<real_type> mode_combination;

        mode_dimensions[0] = nhilb;

        size_type counter;
        std::vector<size_type> nskip(nblocks);
        {
            counter =0;
            for(size_type i=0; i<nblocks; ++i)
            {
                nskip[i] = counter;
                size_type nmaxt = wk[nskip[i]] > 2*wc ? nmaxlargefreq : nmax3;
                mode_combination.set_wk(wk.begin()+counter, wk.begin()+counter + partitions[i]); 
                mode_combination.set_gk(gk[0].begin()+counter, gk[0].begin()+counter + partitions[i]); 
                size_type nstates = mode_combination.construct_basis_topology(ecut, nmaxt);

                real_type ecut2 = ecut;
                size_type nspfmax2 = nspf_lower < nspf ? nspf : nspf_lower;
                size_t nmaxtempo =  nmaxt;
                while(nstates < nspfmax2+1)
                {
                    ecut2 += 0.1*ecut;
                    ++nmaxtempo;
                    nstates = mode_combination.construct_basis_topology(ecut2, nmaxtempo);
                }
                nmodes[i] = nstates;
                mode_dimensions[i+1] = nstates;
                counter += partitions[i];
            }
        }

        std::cerr << nthreads << std::endl;
        blas_set_num_threads(nthreads);

        //now we build the topology tree for 
        ntree<size_type> topology{};    topology.insert(1);
        topology().insert(mode_dimensions[0]);        topology()[0].insert(mode_dimensions[0]);
        topology().insert(mode_dimensions[0]);

        std::cerr << "building subtree" << std::endl;
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

        std::cerr << "topology built" << std::endl;
        std::cerr << "building subtree" << std::endl;
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


        std::cerr << "capacity built" << std::endl;

        std::cerr << std::setprecision(16) << std::endl;

        //now we can construct our initial ttns representation of the wavefunction
        httensor<complex_type, backend_type> A(topology, capacity);     
        std::cerr << "A constructed" << std::endl;

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
            H.bind(ops::dense_matrix_operator<complex_type, backend_type>{Hcoup}, sigma_z_indices, 0); 
        }

        for(size_type i=0; i<nblocks; ++i)
        {   
            size_type nmaxt = wk[nskip[i]] > 2*wc ? nmaxlargefreq : nmax3;
            //set up the required mode information
            mode_combination.set_wk(wk.begin()+nskip[i], wk.begin()+nskip[i] + partitions[i]); 
            mode_combination.set_gk(gk[0].begin()+nskip[i], gk[0].begin()+nskip[i] + partitions[i]); 
            size_type nstates = mode_combination.construct_basis_topology(ecut, nmaxt);
            real_type ecut2 = ecut;

            size_type nspfmax2 = nspf_lower < nspf ? nspf : nspf_lower;
            size_t nmaxtempo =  nmaxt;
            while(nstates < nspfmax2+1)
            {
                ecut2 += 0.1*ecut;
                ++nmaxtempo;
                nstates = mode_combination.construct_basis_topology(ecut2, nmaxtempo);
            }

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
        std::cerr << "sop hamiltonian constructed" << std::endl;

        //and the operator we are computing the value of
        std::vector<ops::dense_matrix_operator<complex_type, backend_type>> mops(4);
        mops[0] = ops::dense_matrix_operator<complex_type, backend_type>(Stot);
        mops[1] = ops::dense_matrix_operator<complex_type, backend_type>(Sx/(1.0*nspins));
        mops[2] = ops::dense_matrix_operator<complex_type, backend_type>(Sy/(1.0*nspins));
        mops[3] = ops::dense_matrix_operator<complex_type, backend_type>(Sz/(1.0*nspins));

        size_t nsteps = static_cast<size_t>(tmax/dt)+50;
    
        std::vector<linalg::vector<real_type>> SzSz(4);
        for(size_type i = 0; i  < 4; ++i){SzSz[i].resize(nsteps);  SzSz[i].fill_zeros();}
        
        std::mt19937 rng(seed);

        ttns::two_site_variations<complex_type, backend_type> twosite;
        std::vector<size_t> state_prev(nblocks, 0);
        //evaluate the Sz1Szk correlation functions.  We need to sample over identity for the Sz1 state
        for(size_type traj = 0; traj < 1; ++traj)//nhilb/2; ++ traj)
        {
            //initialise the wavefunction in the ground state of the bosonic system and in the |0> state of the electronic system
            for(auto& c : A)
            {
                if(c.is_leaf())
                {
                    
                    if(c.leaf_index() == 0)
                    {
                        for(size_type i=0; i<c().size(0); ++i){for(size_type j=0; j<c().size(1); ++j){c()(i, j) = (i == j ? 1.0 : 0.0);}}
                    }
                    else
                    {
                        linalg::matrix<complex_type, backend_type> ct(c().shape(1), c().shape(0));

                        //generate haar random variables
                        for(size_type j=0; j<c().shape(1); ++j){ct(j, 0) = (0 == j ? 1.0 : 0.0);}
                       
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
            std::cerr << "tensor initialised" << std::endl;

            //now we set up the tdvp integrator object
            subspace_expansion_projector_splitting_integrator<complex_type, backend_type> tdvp(A, H, 24, krylov_tolerance, nthreads);
            std::cerr << "integrator initialised" << std::endl;
            //auto& mel = tdvp.mel();
            
            tdvp.dt() = dt;
            tdvp.spawning_threshold() = spawning_parameter;
            tdvp.unoccupied_threshold() = unoccupied_threshold;
            tdvp.minimum_unoccupied() = minimum_unoccupied;
            run(A, tdvp, H, mops, SzSz, tmax, std::cout, true);
        }
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        return 1;
    }

}




