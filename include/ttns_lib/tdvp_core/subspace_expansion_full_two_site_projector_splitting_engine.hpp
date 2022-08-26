#ifndef HTUCKER_SUBSPACE_EXPANSION_FULL_TWO_SITE_PROJECTOR_SPLITTING_ENGINE_HPP
#define HTUCKER_SUBSPACE_EXPANSION_FULL_TWO_SITE_PROJECTOR_SPLITTING_ENGINE_HPP

#define TIMING

#include <random>

#include "tdvp_algorithm_engine.hpp"

#include <linalg/utils/omp.hpp>

#include <utils/krylov_integrator.hpp>
#include "projector_splitting_evolution_functions.hpp"
#include "two_site_energy_variations.hpp"

namespace ttns
{


template <typename T, typename backend>
class subspace_expansion_full_two_site_projector_splitting_engine : public tdvp_algorithm_base<subspace_expansion_full_two_site_projector_splitting_engine, T, backend>
{
protected:
    using twosite = two_site_variations<T, backend>;
    using base_type = tdvp_algorithm_base<subspace_expansion_full_two_site_projector_splitting_engine, T, backend>;
    using vec_type = typename base_type::vec_type;
    using mat_type = typename base_type::mat_type;
    using triad_type = typename base_type::triad_type;
    
    using opdata_type = typename base_type::opdata_type;

    using operator_type = typename base_type::operator_type;

    using size_type = typename base_type::size_type;
    using real_type = typename base_type::real_type;

    using integ_type = krylov_integrator<T, backend>;

    using hnode = typename base_type::hnode;
    using hdata = typename base_type::hdata;
    using opnode = typename base_type::opnode;
    using matnode = typename base_type::matnode;
    using dmat_type = typename base_type::dmat_type;

    friend base_type;
public:
    subspace_expansion_full_two_site_projector_splitting_engine() : base_type(), m_twosite(), m_krylov_dim(0), m_num_threads(1), m_dt(0), m_t(0), m_coeff(1), m_spawning_threshold(-1), m_unoccupied_threshold(-1), m_minimum_unoccupied(0) {}
    subspace_expansion_full_two_site_projector_splitting_engine(const httensor<T, backend>& A, const operator_container<T, backend>& ham, size_type krylov_dim = 4, real_type krylov_threshold = 1e-12, size_type num_threads = 1, size_type seed = 0)  : base_type(), m_twosite(seed), m_num_threads(1), m_spawning_threshold(-1), m_unoccupied_threshold(-1), m_minimum_unoccupied(0)
    {
        CALL_AND_HANDLE(initialise(A, ham, krylov_dim, krylov_threshold, num_threads), "Failed to construct subspace_expansion_full_two_site_projector_splitting_engine.");
    }   
    subspace_expansion_full_two_site_projector_splitting_engine(const subspace_expansion_full_two_site_projector_splitting_engine& o) = default;
    subspace_expansion_full_two_site_projector_splitting_engine(subspace_expansion_full_two_site_projector_splitting_engine&& o) = default;

    subspace_expansion_full_two_site_projector_splitting_engine& operator=(const subspace_expansion_full_two_site_projector_splitting_engine& o) = default;
    subspace_expansion_full_two_site_projector_splitting_engine& operator=(subspace_expansion_full_two_site_projector_splitting_engine&& o) = default;
    
    void initialise(const httensor<T, backend>& A, const operator_container<T, backend>& ham, size_type krylov_dim = 4, real_type krylov_threshold = 1e-12, size_type num_threads = 1)
    {
        try
        {
            ASSERT(num_threads > 0, "Invalid number of threads.");
            ASSERT(A.is_orthogonalised(), "The input hierarchical tucker tensor must have been orthogonalised.");

            CALL_AND_HANDLE(clear(), "Failed to clear the projector_splitting_intgrator.");

            m_num_threads = num_threads;

            //set up openmp environment
            omp_set_dynamic(0);         omp_set_num_threads(m_num_threads);
            mkl_set_dynamic(0);         blas_set_num_threads(m_num_threads);
            
            //set the krylov dimension
            m_krylov_dim =  krylov_dim;

            size_type maxcapacity = 0;  size_type maxnmodes = 0;
            size_type mtwosite_size = 0; size_type mtwosite_capacity = 0;
            for(const auto& a : A)
            {
                size_type capacity = a().capacity();    if(capacity > maxcapacity){maxcapacity = capacity;}
                size_type nmodes = a().nmodes();    if(nmodes > maxnmodes){maxnmodes = nmodes;}

                if(!a.is_root())
                {
                    for(size_type mode = 0; mode < nmodes; ++mode)
                    {
                        auto aptens_c = a.parent()().as_rank_3(mode, true);
                        auto aptens_s = a.parent()().as_rank_3(mode);
                        size_type c2s = a().max_dimen()*aptens_c.shape(0)*aptens_c.shape(1);
                        size_type s2s = a().max_dimen()*aptens_s.shape(0)*aptens_s.shape(1);

                        if(c2s > mtwosite_capacity){mtwosite_capacity = c2s;}
                        if(s2s > mtwosite_size){mtwosite_size = s2s;}
                    }
                }
            }

            CALL_AND_HANDLE(m_twosite_energy.reallocate(mtwosite_capacity), "Failed to allocate two site energy object.");
            CALL_AND_HANDLE(m_twosite_energy.resize(1, mtwosite_size), "Failed to allocate two site energy object.");
            CALL_AND_HANDLE(m_twosite_temp.reallocate(mtwosite_capacity), "Failed to allocate two site energy object.");
            CALL_AND_HANDLE(m_twosite_temp.resize(1, mtwosite_size), "Failed to allocate two site energy object.");

            size_type max_two_site_energy_terms = 0;

            for(const auto& h : ham.op())
            {
                size_type two_site_energy_terms = twosite::get_nterms(h());
                if(two_site_energy_terms > max_two_site_energy_terms){max_two_site_energy_terms = two_site_energy_terms;}
            }

            CALL_AND_HANDLE(m_2s_1.resize(max_two_site_energy_terms), "Failed to resize two site spf buffer.");
            CALL_AND_HANDLE(m_2s_2.resize(max_two_site_energy_terms), "Failed to resize two site mf buffer.");

            for(size_type i = 0; i < max_two_site_energy_terms; ++i)
            {
                CALL_AND_HANDLE(m_2s_1[i].reallocate(maxcapacity), "Failed to reallocate two site spf buffer.");
                CALL_AND_HANDLE(m_2s_2[i].reallocate(maxcapacity), "Failed to reallocate two site mf buffer.");
            }

            m_inds.resize(maxnmodes);
            m_dim.resize(maxnmodes);

            m_rvec.reallocate(maxcapacity);
            m_trvec.reallocate(maxcapacity);
            m_trvec2.reallocate(maxcapacity);

            CALL_AND_HANDLE(m_integ.resize(krylov_dim, maxcapacity), "Failed to initialise the krylov subspace engine object.");
            m_integ.error_tolerance() = krylov_threshold;

            base_type::initialise_base(A);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise the projector_spliting_engine object.");
        }
    }

    void clear()
    {
        try
        {
            CALL_AND_RETHROW(base_type::clear_base());
            for(size_type i = 0; i < m_2s_1.size(); ++i)
            {
                CALL_AND_HANDLE(m_2s_1[i].clear(), "Failed to clear the rvec object.");
                CALL_AND_HANDLE(m_2s_2[i].clear(), "Failed to clear the rvec object.");
            }
            CALL_AND_HANDLE(m_inds.clear(), "Failed to clear temporary inds array.");
            CALL_AND_HANDLE(m_integ.clear(), "Failed to clear the krylov subspace engine.");
            CALL_AND_HANDLE(m_rvec.clear(), "Failed to clear the rvec object.");
            CALL_AND_HANDLE(m_trvec.clear(), "Failed to clear the rvec object.");
            CALL_AND_HANDLE(m_trvec2.clear(), "Failed to clear the rvec object.");
            CALL_AND_HANDLE(m_2s_1.clear(), "Failed to clear the rvec object.");
            CALL_AND_HANDLE(m_2s_2.clear(), "Failed to clear the rvec object.");
            m_krylov_dim = 0;
            m_num_threads = 0;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear the projector_spliting_engine object.");
        }
    }

    void advance_half_step(){m_t += m_dt/2.0;}

    T& coefficient(){return m_coeff;}
    const T& coefficient() const{return m_coeff;}

    real_type& t(){return m_t;}
    const real_type& t() const{return m_t;}

    real_type& dt(){return m_dt;}
    const real_type& dt() const{return m_dt;}

    real_type& spawning_threshold(){return m_spawning_threshold;}
    const real_type& spawning_threshold() const{return m_spawning_threshold;}

    real_type& unoccupied_threshold(){return m_unoccupied_threshold;}
    const real_type& unoccupied_threshold() const{return m_unoccupied_threshold;}

    size_type& minimum_unoccupied(){return m_minimum_unoccupied;}    
    const size_type& minimum_unoccupied() const {return m_minimum_unoccupied;}    

    template <typename Arg>
    void set_rng(const Arg& rng){m_twosite.set_rng(rng);}

protected:
    size_type nbuffers() const
    {
#ifdef PARALLELISE_FOR_LOOPS
        return m_num_threads;
#else
        return 1;
#endif
    }

    size_type get_n_unoccupied(const dmat_type& pops)
    {
        size_type nunocc = 0;
        for(size_type i = 0; i < pops.size(); ++i)
        {
            if(pops(i, i) < m_unoccupied_threshold){++nunocc;}
        }
        return nunocc;
    }

    //TODO: Add in the additional terms that are present when we aren't applying a projector

    //perform the subspace expansion as we are moving down a tree.  This requires us to evaluate the optimal functions to add 
    //into A2.  For A1 they will be overwriten by the r matrix in the next step so we just 
    //here the A1 and A2 tensors must be left and right orthogonal respectively with the non-orthogonal component stored in r
    bool subspace_expansion_down(hnode& A1, hnode& A2, mat_type& r, const dmat_type& pops, opnode& h, const operator_type& op)
    {
        try
        {
            size_type mode = A1.child_id();

            //first check if the mode we are interested in has the capacity to be expanded
            if(A2().dim(mode) >= A2().max_dim(mode) || A1().hrank() >= A1().max_hrank()){return false;}

            //get the maximum number of elements that we can add to the tensor
            size_type max_add = A2().max_dim(mode) - A2().dim(mode);
            bool subspace_expanded = false;

            //determine if we should attempt subspace expansion.
            
            auto A2tens = A2().as_rank_3(mode);
            size_type max_dimension = A2tens.shape(0)*A2tens.shape(2);

            //if the current bond is equal in size to the product of all other bonds then we aren't able to do any
            //then the two site object has rank to max_dimension and we won't be able to expand the onesite tensors
            //as they already fully capture the rank of the two-site tensor.
            if(A2tens.shape(1) >= max_dimension){return false;}

            size_type n_unocc = get_nunoccupied(pops);
            if(n_unocc >= m_minimum_unoccupied){return false;}

            //get the maximum number of elements that we can add
            size_type max_add2 = max_dimension - A2tens.shape(1);

            //we can add min(max_add, max_add2) elements
            max_add = max_add < max_add2 ? max_add : max_add2;

            //now determine the number of unoccupied functions, and using this determine the minimum number of terms we need to add
            //size_type n_unocc = get_n_unoccupied(pops);
            size_type required_terms = 0;

            //and limit the required number of terms to the maximum number of terms we can add
            if(required_terms > max_add){required_terms = max_add;}
    
            //now resize the onesite buffer objects used for evaluation of the two site (projected energy) so that everything is the correct size
            size_type nterms = twosite::get_nterms(h()); 
            ASSERT(m_2s_1.size() >= nterms, "Unable to store temporaries needed for computing the two-site energy.  The buffers are not sufficient.");
            CALL_AND_HANDLE(resize_two_site_energy_buffers(nterms, A1(), A2(), mode), "Failed to resize the objects needed to compute the two-site energy.");
            CALL_AND_HANDLE(m_inds.resize(nterms), "Failed to resize indices array.");
            twosite::set_indices(h(), m_inds); 

            //compute the one site objects that are used to compute the two site projected energy
            CALL_AND_HANDLE(twosite::construct_two_site_energy_terms_lower(A1, h, op, m_2s_1, base_type::m_HA, base_type::m_temp, m_inds, r, true), 
                            "Failed to construct the component of the two site energy acting on the lower site.");
            CALL_AND_HANDLE(twosite::construct_two_site_energy_terms_upper(A2, h, m_2s_2, base_type::m_HA, base_type::m_temp, base_type::m_temp2, m_inds, true), 
                            "Failed to construct the component of the two site energy acting on the upper site.");

                    
            //explicitly compute the two site energy object.
            CALL_AND_HANDLE(m_twosite_energy.resize(A1().shape(0), A2tens.shape(0)*A2tens.shape(2)), "Failed to resize two site energy object.");
            CALL_AND_HANDLE(m_twosite_temp.resize(A1().shape(0), A2tens.shape(0)*A2tens.shape(2)), "Failed to resize two site energy object.");
            CALL_AND_HANDLE(twosite::construct_two_site_energy(m_2s_1, m_2s_2, m_twosite_temp, m_twosite_energy), "Failed to create two site energy object.");

            CALL_AND_HANDLE(m_svd(m_twosite_energy, m_S, m_U, m_V), "Failed to compute svd of two site matrix.");
            
            size_type nadd = 0;
            for(size_type i = 0; i < m_S.size(); ++i)
            {
                if(m_S(i, i)*(m_dt*m_dt)/4.0 > m_spawning_threshold){++nadd;}
            }
            //if(nadd == 0){++nadd;}
            if(nadd > max_add){nadd = max_add;}
            if(nadd != 0)
            {
                //expand the A1 tensor zero padding
                CALL_AND_HANDLE(twosite::expand_tensor(A1(), base_type::m_temp[0], A1.nmodes(), nadd, m_dim), "Failed to expand A1 tensor.");

                //expand the r-matrix
                CALL_AND_HANDLE(twosite::expand_matrix(r, base_type::m_temp[0], nadd), "Failed to expand R matrix.");

                //expand the A2 tensor padding with the right singular vectors
                CALL_AND_HANDLE(twosite::expand_tensor(A2(), base_type::m_temp[0], mode, m_V, nadd, m_dim), "Failed to expand A2 tensor.");

                //resize all of the working buffers. 
                CALL_AND_HANDLE(base_type::resize_working_buffers(A2().shape(0), A2().shape(1)), "Failed to resize working arrays.");

                //resize the Hamiltonian object stored at the child node to the correct size.  Here we don't care about the values 
                //stored in these matrices as they will be updated before they are used for anything.
                CALL_AND_HANDLE(h().resize_matrices(r.shape(0), r.shape(1)), "Failed to resize Hamiltonian matrices.");

                subspace_expanded = true;
            }
                          
            return subspace_expanded;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to perform the subspace expansion when traversing down the tree.");
        }
    }

    //here the A1 and A2 tensors must be left and right orthogonal respectively with the non-orthogonal component stored in r
    bool subspace_expansion_up(hnode& A1, hnode& A2, mat_type& r, const dmat_type&  pops, opnode& h, const operator_type& op)
    {
        try
        {
            size_type mode = A1.child_id();

            //first check if the mode we are interested in has the capacity to be expanded
            if(A1().hrank() >= A1().max_hrank()){return false;}

            //get the maximum number of elements that we can add to the tensor
            size_type max_add = A1().max_hrank() - A1().hrank();
            bool subspace_expanded = false;

            //determine if we should attempt subspace expansion.
            
            auto A2tens = A2().as_rank_3(mode);
            size_type max_dimension = A1().shape(0);

            //if the current bond is equal in size to the product of all other bonds then we aren't able to do any
            //then the two site object has rank to max_dimension and we won't be able to expand the onesite tensors
            //as they already fully capture the rank of the two-site tensor.
            if(A1().shape(1) > max_dimension){return false;}

            size_type n_unocc = get_nunoccupied(pops);
            if(n_unocc >= m_minimum_unoccupied){return false;}

            //get the maximum number of elements that we can add
            size_type max_add2 = max_dimension - A1().shape(1);

            //we can add min(max_add, max_add2) elements
            max_add = max_add < max_add2 ? max_add : max_add2;

            //now determine the number of unoccupied functions, and using this determine the minimum number of terms we need to add
            //size_type n_unocc = get_n_unoccupied(pops);
            size_type required_terms = 0;

            //and limit the required number of terms to the maximum number of terms we can add
            if(required_terms > max_add){required_terms = max_add;}
    
            //now resize the onesite buffer objects used for evaluation of the two site (projected energy) so that everything is the correct size
            size_type nterms = twosite::get_nterms(h()); 

            ASSERT(m_2s_1.size() >= nterms, "Unable to store temporaries needed for computing the two-site energy.  The buffers are not sufficient.");
            CALL_AND_HANDLE(resize_two_site_energy_buffers(nterms, A1(), A2(), mode), "Failed to resize the objects needed to compute the two-site energy.");
            CALL_AND_HANDLE(m_inds.resize(nterms), "Failed to resize indices array.");


            twosite::set_indices(h(), m_inds); 

            //compute the one site objects that are used to compute the two site projected energy
            CALL_AND_HANDLE(twosite::construct_two_site_energy_terms_upper(A2, h, m_2s_2, base_type::m_HA, base_type::m_temp, base_type::m_temp2, m_inds, true), 
                            "Failed to construct the component of the two site energy acting on the upper site.");
            CALL_AND_HANDLE(twosite::construct_two_site_energy_terms_lower(A1, h, op, m_2s_1, base_type::m_HA, base_type::m_temp, m_inds, r, true), 
                            "Failed to construct the component of the two site energy acting on the lower site.");
                    
            //explicitly compute the two site energy object.
            CALL_AND_HANDLE(m_twosite_energy.resize(A1().shape(0), A2tens.shape(0)*A2tens.shape(2)), "Failed to resize two site energy object.");
            CALL_AND_HANDLE(m_twosite_temp.resize(A1().shape(0), A2tens.shape(0)*A2tens.shape(2)), "Failed to resize two site energy object.");

            CALL_AND_HANDLE(twosite::construct_two_site_energy(m_2s_1, m_2s_2, m_twosite_temp, m_twosite_energy), "Failed to create two site energy object.");
            std::cerr << m_twosite_energy << std::endl;
            CALL_AND_HANDLE(m_svd(m_twosite_energy, m_S, m_twosite_temp, m_V), "Failed to compute svd of two site matrix.");

            m_U = linalg::trans(m_twosite_temp);
            ////need to figure out the which of the 
            //std::cerr << "exact" << std::endl;
            //std::cerr << m_S << std::endl;

            //linalg::diagonal_matrix<T> S;
            //linalg::matrix<T> vecs;
            //size_type neigs = 4;

            //size_type maxkrylov_dim = m_krylov_dim;
            //m_rvec = m_U[0];
            //if(m_rvec.size() < maxkrylov_dim){maxkrylov_dim = m_rvec.size();}

            //if(!A1.is_leaf())
            //{
            //    std::cerr << m_U << std::endl;
            //}
            //
            //bool mconjm = false;
            //real_type scale_factor = 1;
            //CALL_AND_HANDLE(m_integ.eigs(m_rvec, S, vecs, scale_factor, neigs, maxkrylov_dim, 1e-4, m_twosite, m_2s_1, m_2s_2, m_trvec, m_trvec2, base_type::m_temp[0], mconjm), "Failed to compute sparse svd.");
            //std::cerr << "krylov1" << std::endl;
            //for(size_type si = 0; si < S.size(); ++si)
            //{
            //    std::cerr << std::sqrt(std::real(S(si, si))) << " " << S(si, si) << std::endl;
            //}

            //if(!A1.is_leaf())
            //{
            //    std::cerr << vecs << std::endl;
            //}
            //std::cerr << vecs[1] << std::endl;
            //m_rvec = m_V[0];
            //size_type maxkrylov_dim = m_krylov_dim;
            //if(m_rvec.size() < maxkrylov_dim){maxkrylov_dim = m_rvec.size();}
            //bool mconjm = true;
            //CALL_AND_HANDLE(m_integ.eigs(m_rvec, S, vecs, scale_factor, neigs, maxkrylov_dim, 1e-4, m_twosite, m_2s_1, m_2s_2, m_trvec, m_trvec2, base_type::m_temp[0], mconjm), "Failed to compute sparse svd.");
            //std::cerr << "krylov 2" << std::endl;
            //for(size_type si = 0; si < S.size(); ++si)
            //{
            //    std::cerr << std::sqrt(std::real(S(si, si))) << " " << S(si, si) << std::endl;
            //}
            size_type nadd = 0;
            for(size_type i = 0; i < m_S.size(); ++i)
            {
                if(m_S(i, i)*(m_dt*m_dt)/4.0 > m_spawning_threshold){++nadd;}
            }

            //if(nadd == 0){++nadd;}
            if(nadd > max_add){nadd = max_add;}
            //here we need to compute the eigenstates of the 
            if(nadd != 0)
            {
                //expand the A2 tensor padding with the right singular vectors
                CALL_AND_HANDLE(twosite::expand_tensor(A2(), base_type::m_temp[0], mode, nadd, m_dim), "Failed to expand A2 tensor.");

                //expand the r-matrix
                CALL_AND_HANDLE(twosite::expand_matrix(r, base_type::m_temp[0], nadd), "Failed to expand R matrix.");

                //expand the A1 tensor around the index pointing to the root zero padding
                CALL_AND_HANDLE(twosite::expand_tensor(A1(), base_type::m_temp[0], A1.nmodes(), m_U, nadd, m_dim), "Failed to expand A1 tensor.");

                //resize all of the working buffers. 
                CALL_AND_HANDLE(base_type::resize_working_buffers(A1().shape(0), A1().shape(1)), "Failed to resize working arrays.");

                //resize the Hamiltonian object stored at the child node to the correct size.  Here we don't care about the values 
                //stored in these matrices as they will be updated before they are used for anything.
                CALL_AND_HANDLE(h().resize_matrices(r.shape(0), r.shape(1)), "Failed to resize Hamiltonian matrices.");

                subspace_expanded = true;
            }
                          
            return subspace_expanded;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to perform the subspace expansion when traversing up the tree.");
        }

    }


    void update_node_tensor(hnode& A, opnode& h, operator_type& op)
    {                    
        if(!A.is_leaf())
        {
#ifdef PARALLELISE_FOR_LOOPS
            CALL_AND_HANDLE
            (
                base_type::m_nh_evals += m_integ(A().as_matrix(), m_dt/2.0, m_coeff, ceb, h, A.hrank(), A.dims(), base_type::m_HA, base_type::m_temp, base_type::m_temp2),
                "Failed to evolve the branch coefficient matrix."
            );
#else
            CALL_AND_HANDLE
            (
                base_type::m_nh_evals += m_integ(A().as_matrix(), m_dt/2.0, m_coeff, ceb, h, A.hrank(), A.dims(), base_type::m_HA[0], base_type::m_temp[0], base_type::m_temp2[0]),
                "Failed to evolve the branch coefficient matrix."
            );
#endif
        }
        else
        {
            //if we are at a leaf node we update its child operators.  This is simply done by calling the update function
            //for its hamiltonia operator object
            CALL_AND_HANDLE(op.update(A.leaf_index(), m_t, m_dt/2.0), "Failed to update primitive Hamiltonian object.");

#ifdef PARALLELISE_FOR_LOOPS
            CALL_AND_HANDLE
            (
                base_type::m_nh_evals += m_integ(A().as_matrix(), m_dt/2.0, m_coeff, cel, h, op, base_type::m_HA, base_type::m_temp),
                "Failed to evolve the leaf coefficient matrix."
            );
#else
            CALL_AND_HANDLE
            (
                base_type::m_nh_evals += m_integ(A().as_matrix(), m_dt/2.0, m_coeff, cel, h, op, base_type::m_HA[0], base_type::m_temp[0]),
                "Failed to evolve the leaf coefficient matrix."
            );
#endif
        }
    }

    void update_R_tensor(matnode& r, opnode& h)   
    {
#ifdef PARALLELISE_FOR_LOOPS
        CALL_AND_HANDLE(m_integ(r(), -m_dt/2.0, m_coeff, fha, h, base_type::m_HA, base_type::m_temp), "Failed to time evolve the r matrix backwards in time.");
#else   
        CALL_AND_HANDLE(m_integ(r(), -m_dt/2.0, m_coeff, fha, h, base_type::m_HA[0]), "Failed to time evolve the r matrix backwards in time.");
#endif
    }

protected:
    void resize_two_site_energy_buffers(size_type n, const hdata& a1, const hdata& a2, size_type mode)
    {
        auto atens = a2.as_rank_3(mode);
        for(size_type _n = 0; _n < n; ++_n)
        {
            CALL_AND_HANDLE(m_2s_1[_n].resize(a1.shape(0), a1.shape(1)), "Failed to reshape two-site energy buffer object.");
            CALL_AND_HANDLE(m_2s_2[_n].resize(atens.shape(0), atens.shape(1), atens.shape(2)), "Failed to reshape two-site energy buffer object.");
        }
    }

    size_type get_nunoccupied(const dmat_type&  pops)
    {
        size_type nunocc = 0;
        for(size_type i = 0; i  < pops.size(); ++i)
        {
            if(pops(i, i)/pops(0, 0) < m_unoccupied_threshold){++nunocc;}
        }
        return nunocc;
    }

protected:
    full_hamiltonian_action<T, backend> fha;        //used for evolving the r matrix under the full Hamiltonian
    coefficient_evolution_leaf<T, backend> cel;     //used for evolving the coefficient tensors of leaf nodes
    coefficient_evolution_branch<T, backend> ceb;   //used for evolving the coefficient tensors of branch nodes
    two_site_variations<T, backend> m_twosite;

    vec_type m_rvec;
    mat_type m_trvec;
    mat_type m_trvec2;

    //the krylov subspace engine
    integ_type m_integ;

    std::vector<size_type> m_dim;
    size_type m_krylov_dim;
    size_type m_num_threads;

    //add in a second set of indices
    linalg::vector<size_type> m_inds;
    triad_type m_2s_1;
    std::vector<linalg::tensor<T, 3, backend>> m_2s_2;

    real_type m_dt;
    real_type m_t;
    T m_coeff;
    real_type m_spawning_threshold;
    real_type m_unoccupied_threshold;
    size_type m_minimum_unoccupied;

    singular_value_decomposition<mat_type, true> m_svd;
    mat_type m_twosite_energy;
    mat_type m_twosite_temp;
    mat_type m_U;
    dmat_type m_S;
    mat_type m_V;
};  //class projector_splitting_engine

}   //namespace ttns

#endif  //HTUCKER_SUBSPACE_PROJECTOR_SPLITTING_ENGINE_HPP//

