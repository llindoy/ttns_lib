#ifndef MODE_DATA_HPP
#define MODE_DATA_HPP

#include <vector>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <utility>
#include <complex>
#include <unordered_set>
#include <tuple>
#include <random>

#include <linalg/linalg.hpp>
#include <linalg/decompositions/eigensolvers/eigensolver.hpp>
#include <ttns_lib/ttns.hpp>

#include "system_bath_coupling_interaction_picture_operator.hpp"

template <typename T>
class harmonic_mode_combination
{
protected:
    using real_type = T;
    using complex_type = ttns::complex<T>;
    std::vector<T> m_wk;
    std::vector<T> m_gk;
    std::vector<size_t> m_max_states;

    linalg::matrix<size_t> m_nk;
    linalg::matrix<int32_t> m_nkp1;
    linalg::matrix<int32_t> m_nkm1;

    bool m_w_set;
    bool m_c_set;

    bool m_ind_set;
public:
    harmonic_mode_combination() :  m_w_set(false), m_c_set(false), m_ind_set(false) {}
    harmonic_mode_combination(size_t n) : m_wk(n), m_gk(n), m_w_set(false), m_c_set(false), m_ind_set(false) {}

    template <typename R1, typename R2>
    harmonic_mode_combination(const R1& w, const R2& c) : m_w_set(true), m_c_set(true), m_ind_set(false)
    {
        ASSERT(w.size() = c.size(), "Failed to initialise harmonic mode combination object from two arrays.  The arrays are not the same size.");
        m_wk.resize(w.size());
        m_gk.resize(c.size());
        for(size_t i = 0; i < w.size(); ++i)
        {
            m_wk[i] = w[i];
            m_gk[i] = c[i];
        }
    }

    template <class R1, class R2>
    harmonic_mode_combination(R1 w1, R1 w2, R2 c1, R2 c2) : m_w_set(true), m_c_set(true), m_ind_set(false)
    {
        std::ptrdiff_t nw = w2 - w1;
        std::ptrdiff_t nc = c2 - c1;

        ASSERT(nw == nc, "Failed to initialise harmonic mode combination object from two sets of iterable objects.  The two sets do not have the same size.");
        m_wk.resize(nw);
        m_gk.resize(nw);
        
        size_t c = 0;
        for(R1 a = w1; a != w2; ++a, ++c){m_wk[c] = *a;}
    
        c = 0;
        for(R2 a = c1; a != c2; ++a, ++c){m_gk[c] = *a;}
    }

    void resize(size_t n) 
    {
        m_w_set = false;    m_c_set = false;    m_ind_set = false;
        CALL_AND_HANDLE(m_wk.resize(n), "Failed to resize harmonic mode combination object.  Failed to resize the frequency array.");
        CALL_AND_HANDLE(m_gk.resize(n), "Failed to resize harmonic mode combination object.  Failed to resize the coupling array.");
        CALL_AND_HANDLE(m_nk.clear(), "Failed to resize harmonic mode combination object.  Failed to clear the topology array.");
        CALL_AND_HANDLE(m_nkp1.clear(), "Failed to resize harmonic mode combination object.  Failed to clear the topology array.");
        CALL_AND_HANDLE(m_nkm1.clear(), "Failed to resize harmonic mode combination object.  Failed to clear the topology array.");
    }

    //functions for setting the inputs necessary
    template <typename R1>
    void set_wk(const R1& w)
    {
        if(w.size() != m_wk.size())
        {
            CALL_AND_HANDLE(resize(w.size()), "Failed to set the frequency array of the harmonic mode combination object.");
        }
        for(size_t i=0; i<w.size(); ++i){m_wk[i] = w[i];}
        m_w_set = true;
    }

    template <typename R1>
    void set_wk(R1 w1, R1 w2)
    {
        std::ptrdiff_t nw = w2 - w1;
        if(nw != static_cast<std::ptrdiff_t>(m_wk.size()))
        {
            CALL_AND_HANDLE(resize(nw), "Failed to set the frequency array of the harmonic mode combination object.");
        }
        
        size_t c = 0;
        for(R1 a = w1; a != w2; ++a, ++c){m_wk[c] = *a;}
        m_w_set = true;
    }

    template <typename R1>
    void set_gk(const R1& c)
    {
        if(c.size() != m_gk.size())
        {
            CALL_AND_HANDLE(resize(c.size()), "Failed to set the coupling array of the harmonic mode combination object.");
        }
        for(size_t i=0; i<c.size(); ++i){m_gk[i] = c[i];}
        m_c_set = true;
    }

    template <typename R1>
    void set_gk(R1 c1, R1 c2)
    {
        std::ptrdiff_t nc = c2 - c1;
        if(nc != static_cast<std::ptrdiff_t>(m_gk.size()))
        {
            CALL_AND_HANDLE(resize(nc), "Failed to set the coupling array of the harmonic mode combination object.");
        }
        
        size_t c = 0;
        for(R1 a = c1; a != c2; ++a, ++c){m_gk[c] = *a;}
        m_c_set = true;
    }


    //functions for computing the coupling topology required
    size_t construct_basis_topology(T ecut, size_t nmax)
    {
        ASSERT(m_w_set, "Failed to compute the basis topology.   The frequencies of the harmonic modes have not been set.");
        size_t nstates = compute_nstates(ecut, nmax);
        size_t N = m_wk.size();
        m_max_states.resize(N);
        for(size_t i=0; i < N; ++i)
        {
            real_type wi = std::abs(m_wk[i]);
            size_t j;
            for(j=0; j<=static_cast<size_t>(std::ceil(ecut/wi)); ++j){}
            m_max_states[i] = j;
        }

        if(N > 1)
        {
            linalg::vector<size_t> nt(N);
            m_nk.resize(nstates, N);
            populate_state_vector(m_nk, nt, 0, 0, ecut, nmax);
        }
        else
        {
            nstates = nstates > nmax ? nmax : nstates;
            m_nk.resize(nstates, N);
            for(size_t i=0; i < nstates; ++i)
            {
                m_nk(i, 0) = i;
            }
        }

        //now we need to create a vector of pairs useful for the searching ofvalue arrays
        std::vector<std::vector<size_t>> nk(nstates);   
        for(size_t i=0; i < nstates; ++i)
        {   
            nk[i].resize(N);
            for(size_t j=0; j<N; ++j){nk[i][j] = m_nk(i,j);}
        }

    
        //now we construct the list of lowering operators
        m_nkm1.resize(nstates, N);  m_nkm1.fill_value(-1.0);
        m_nkp1.resize(nstates, N);  m_nkp1.fill_value(-1.0);

        std::vector<size_t> nt(N);
        for(size_t i=0; i < nstates; ++i)
        {
            nt = nk[i];
            for(size_t j=0; j<N; ++j)
            {
                if(nt[j] == 0){m_nkm1(i, j) = -1;}
                else
                {
                    nt[j] = nk[i][j] - 1;
                    auto itr = std::lower_bound(nk.begin(), nk.end(), nt, 
                                    [](const std::vector<size_t>& a, const std::vector<size_t>& b)
                                    {
                                        for(size_t ii=0; ii<a.size(); ++ii)
                                        {
                                            if(a[ii] < b[ii]){return true;}
                                            else if(a[ii] > b[ii]){return false;}
                                        }
                                        return false;
                                    }
                                );
                    if(itr == nk.end()){m_nkm1(i, j) = -1;}
                    else if(*itr != nt){m_nkm1(i, j) = -1;}
                    else{m_nkm1(i, j) = std::distance(nk.begin(), itr);}
                }

                //now we attempt to find the +1
                nt[j] = nk[i][j] + 1;
                auto itr = std::lower_bound(nk.begin(), nk.end(), nt, 
                                [](const std::vector<size_t>& a, const std::vector<size_t>& b)
                                {
                                    for(size_t ii=0; ii<a.size(); ++ii)
                                    {
                                        if(a[ii] < b[ii]){return true;}
                                        else if(a[ii] > b[ii]){return false;}
                                    }
                                    return false;
                                }
                            );
                if(itr == nk.end()){m_nkp1(i, j) = -1;}
                else if(*itr != nt){m_nkp1(i, j) = -1;}
                else{m_nkp1(i, j) = std::distance(nk.begin(), itr);}

                nt[j] = nk[i][j];
            }
        }

        m_ind_set = true;
        return nstates;
    }

    size_t compute_nstates(T ecut, size_t nmax)
    {
        ASSERT(m_w_set, "Failed to compute the number of states.   The frequencies of the harmonic modes have not been set.");
        m_max_states.resize(m_wk.size());
        return get_nstates_in_cut(0, ecut, nmax);
    }


    //functions for computing matrices necessary for spin-boson (and polaron transformed spin-boson) dynamics
    template <typename U, typename backend>
    typename std::enable_if<std::is_same<T, U>::value || std::is_same<complex_type, U>::value, void>::type H0(linalg::diagonal_matrix<U, backend>& H)
    {
        ASSERT(m_w_set && m_ind_set, "Failed to construct H0.  Either the topology or frequency arrays have not been set.");
        size_t nstates = m_nk.shape(0);     size_t N = m_nk.shape(1);
        H.resize(nstates, nstates);
        linalg::diagonal_matrix<U> _H(nstates, nstates);
        for(size_t i=0; i<nstates; ++i)
        {
            T val = 0.0;
            for(size_t j = 0; j < N; ++j){val += m_wk[j]*(m_nk(i, j)+0.5-0.5);}
            _H[i] = val;
        }
        H = _H;
    }

    template <typename RNG>
    size_t random_state(T& E, RNG&& rng)
    {
        std::uniform_int_distribution<size_t> dist = std::uniform_int_distribution<size_t>(0, m_nk.shape(0)-1);
        size_t state = dist(rng);

        std::vector<size_t> N(m_nk.shape(1));
        E = 0.0;
        for(size_t i=0; i < m_nk.shape(1); ++i)
        {
            N[i] = m_nk(state, i);
            E += N[i]*m_wk[i];
        }
        return state;
    }


    T get_energy(size_t state)
    {
        T E = 0.0;
        for(size_t i=0; i < m_nk.shape(1); ++i)
        {
            E += m_nk(state, i)*m_wk[i];
        }
        return E;
    }

    T partition_function(const T& beta)
    {
        T Q = 1;
        for(size_t i =0; i < m_nk.shape(1); ++i)
        {
            Q *= 1.0/(1.0 - std::exp(-beta*m_wk[i]));
        }
        return Q;
    }
    
    template <typename U, typename backend>
    typename std::enable_if<std::is_same<T, U>::value || std::is_same<complex_type, U>::value, void>::type H0(linalg::matrix<U, backend>& H)
    {
        ASSERT(m_w_set && m_ind_set, "Failed to construct H0.  Either the topology or frequency arrays have not been set.");
        size_t nstates = m_nk.shape(0);     size_t N = m_nk.shape(1);
        H.resize(nstates, nstates);
        linalg::matrix<U> _H;  _H.resize(nstates, nstates); _H.fill_zeros();
        for(size_t i=0; i<nstates; ++i)
        {
            T val = 0.0;
            for(size_t j = 0; j < N; ++j){val += m_wk[j]*(m_nk(i, j)+0.5-0.5);}
            _H(i, i) = val;
        }
        H = _H;
    }

    //functions for computing matrices necessary for spin-boson (and polaron transformed spin-boson) dynamics
    template <typename U, typename backend>
    typename std::enable_if<std::is_same<T, U>::value || std::is_same<complex_type, U>::value, void>::type Hc(linalg::csr_matrix<U, backend>& H)
    {
        ASSERT(m_w_set && m_ind_set && m_c_set, "Failed to construct Hc.  Not all of the topology, frequency and coupling arrays have been set.");
        size_t nstates = m_nk.shape(0);     size_t N = m_nk.shape(1);
        linalg::csr_matrix<U> _H;
        _H.resize(nstates, nstates);        H.resize(nstates, nstates);

        //we determine the number of non-zeros in the coupling element, while also setting the rowptr array
        auto rowptr = _H.rowptr();   rowptr[0] = 0;
        size_t nnz = 0;
        for(size_t i=0; i<nstates; ++i)
        {
            for(size_t j=0; j<N; ++j)
            {
                nnz += (m_nkm1(i, j) != -1 ? 1 : 0) + (m_nkp1(i, j) != -1 ? 1 : 0);
            }
            rowptr[i+1] = nnz;
        }
        _H.resize(nnz);
        H.resize(nnz);

        //now we can set the colind and buffer values
        size_t counter = 0;
        auto colind = _H.colind();
        auto buffer = _H.buffer();
        for(size_t i=0; i<nstates; ++i)
        {
            for(size_t j=0; j<N; ++j)
            {
                if(m_nkm1(i, j) != -1)
                {
                    T n = static_cast<T>(m_nk(i, j));
                    colind[counter] = m_nkm1(i, j);
                    buffer[counter] = sqrt(n)*m_gk[j];
                    ++counter;
                }
            }

            for(size_t k=0; k<N; ++k)
            {
                size_t j = N-(k+1);
                if(m_nkp1(i, j) != -1)
                {
                    T n = static_cast<T>(m_nk(i, j)+1.0);
                    colind[counter] = m_nkp1(i, j);
                    buffer[counter] = sqrt(n)*m_gk[j];
                    ++counter;
                }
            }
        }
        H = _H;
    }
    
    template <typename U, typename backend>
    typename std::enable_if<std::is_same<T, U>::value || std::is_same<complex_type, U>::value, void>::type Hc(linalg::matrix<U, backend>& H)
    {
        ASSERT(m_w_set && m_ind_set && m_c_set, "Failed to construct Hc.  Not all of the topology, frequency and coupling arrays have been set.");
        size_t nstates = m_nk.shape(0);     size_t N = m_nk.shape(1);
        linalg::matrix<U> _H(nstates, nstates);     _H.fill_zeros();
        H.resize(nstates, nstates);

        for(size_t i=0; i<nstates; ++i)
        {
            for(size_t j=0; j<N; ++j)
            {
                if(m_nkm1(i, j) != -1)
                {
                    T n = static_cast<T>(m_nk(i, j)+0.0);
                    _H(i, m_nkm1(i, j)) = sqrt(n)*m_gk[j];
                }
            }

            for(size_t k=0; k<N; ++k)
            {
                size_t j = N-(k+1);
                if(m_nkp1(i, j) != -1)
                {
                    T n = static_cast<T>(m_nk(i, j)+1.0);
                    _H(i, m_nkp1(i, j)) = sqrt(n)*m_gk[j];
                }
            }
        }
        H = _H;
    }


    template <typename U, typename backend>
    typename std::enable_if<std::is_same<T, U>::value || std::is_same<complex_type, U>::value, void>::type Hc_interact(interaction_picture_sb_operator<U, backend>& H)
    {
        ASSERT(m_w_set && m_ind_set && m_c_set, "Failed to construct Hc.  Not all of the topology, frequency and coupling arrays have been set.");
        CALL_AND_HANDLE(H.initialise(m_wk, m_gk, m_nk, m_nkp1, m_nkm1), "Failed to intiailise operator.");
    }

    //this is still not stable enough
    template <typename backend>
    void form_displacement_operator(linalg::matrix<complex_type, backend>& Hm, const std::vector<complex_type>& _alpha)
    {
        size_t nstates = m_nk.shape(0);     size_t N = m_nk.shape(1);
        std::vector<linalg::matrix<complex_type> > Dk(N);
        linalg::matrix<complex_type> h(nstates, nstates);   h.fill_zeros();

        for(size_t mode=0; mode < N; ++mode)
        {
            complex_type alpha = _alpha[mode];//sqrt(alpha_factor*m_wk[mode])*m_ck[mode]/(m_wk[mode]*m_wk[mode]);
            complex_type nalpha_conj = -conj(alpha);
            real_type abs_alpha = abs(alpha);
            real_type a2 = abs_alpha*abs_alpha;
            real_type expa2 = exp(-a2/2.0);

            size_t ni = (N == 1 ? nstates : m_max_states[mode]);
            if(ni < 2){ni = 2;}

            Dk[mode].resize(ni, ni);

            //compute the displacement matrix using the recursion relations for the elements:
            //  \bra{0}D(\alpha)\ket{0} = exp(-|\alpha|^2/2)
            //  \bra{1}D(\alpha)\ket{1} = exp(-|\alpha|^2/2)(1-|\alpha|^2)
            //  \bra{n}D(\alpha)\ket{n} = \frac{2n-1-|\alpha|^2}{n} \bra{n-1}D(\alpha)\ket{n-1} - \frac{n-1}{n}\bra{n-2}D(\alpha)\ket{n-2}
            //  \bra{n}D(\alpha)\ket{1} = f_n*(n-|\alpha|^2) where
            //      f_1 = exp(-|\alpha|^2/2)
            //      f_n = f_{n-1}*\frac{\alpha}{n}
            //  \bra{n}D(\alpha)\ket{m} = \frac{m+n-1-|\alpha|^2}{\sqrt{mn}} \bra{n-1}D(\alpha)\ket{n-1} - \sqrt{\frac{(n-1)(m-1)}{mn}} \bra{n-2}D(\alpha)\ket{m-2} n > m
            //  and that \bra{m}D(\alpha)\ket{n} = \bra{n}D(-\alpha)\ket{m}^*
            //first populate the second row and column
            Dk[mode](0, 0) = expa2;
            Dk[mode](1, 1) = expa2;
            for(size_t n=2; n < ni; ++n)
            {
                Dk[mode](n, 1) = alpha*Dk[mode](n-1, 1)/sqrt(static_cast<real_type>(n));
            }
            for(size_t m=2; m < ni; ++m)
            {
                Dk[mode](1, m) = nalpha_conj*Dk[mode](1, m-1)/sqrt(static_cast<real_type>(m));
            }

            Dk[mode](1, 1) = expa2*(1.0-a2);
            for(size_t n=2; n<ni; ++n){Dk[mode](n, 1) *= (n-a2);}
            for(size_t m=2; m<ni; ++m){Dk[mode](1, m) *= (m-a2);}

            //Now we populate the diagonals
            Dk[mode](0, 0) = expa2; Dk[mode](1,1) = expa2*(1-a2);
            for(size_t i=2; i < ni; ++i)
            {
                Dk[mode](i, i) = ((2.0*i-1.0-a2) * Dk[mode](i-1, i-1) - (i-1.0)*Dk[mode](i-2, i-2))/static_cast<real_type>(i);
            }
            
            //now populate the first column (all values with m=0)
            for(size_t n=1; n < ni; ++n)
            {
                Dk[mode](n, 0) = alpha/sqrt(static_cast<real_type>(n))*Dk[mode](n-1, 0);
            }

            //and first row (all values with n=0)
            for(size_t m=1; m < ni; ++m)
            {
                Dk[mode](0, m) = nalpha_conj/sqrt(static_cast<real_type>(m))*Dk[mode](0, m-1);
            }

            //now we can compute all values with n > m
            for(size_t d=1; d<ni; ++d)
            {
                for(size_t n=d+2; n < ni; ++n)
                {   
                    size_t m= n-d;
                    Dk[mode](n, m) = (m+n-1.0-a2)/sqrt(static_cast<real_type>(m*n))*Dk[mode](n-1, m-1) - sqrt((m-1.0)*(n-1.0)/(m*n))*Dk[mode](n-2, m-2);
                }
            }
            for(size_t d=1; d<ni; ++d)
            {
                for(size_t m=d+2; m < ni; ++m)
                {   
                    size_t n = m-d;
                    Dk[mode](n, m) = (m+n-1.0-a2)/sqrt(static_cast<real_type>(m*n))*Dk[mode](n-1, m-1) - sqrt((m-1.0)*(n-1.0)/(m*n))*Dk[mode](n-2, m-2);
                }
            }
        }
        for(size_t i=0; i<nstates; ++i)
        {
            for(size_t j=0; j<nstates; ++j)
            {
                h(i, j) = 1.0;
                for(size_t mode=0; mode < N; ++mode)
                {
                    h(i, j) *= Dk[mode](m_nk(i, mode), m_nk(j, mode));
                }
            }
        }
        Hm = h;
    }
   
    template <typename vec_type, typename RNG>
    void psi0_sampled(vec_type&& psi0, const T& beta, RNG&& rng)
    {
        ASSERT(m_w_set && m_ind_set && m_c_set, "Failed to construct Hc.  Not all of the topology, frequency and coupling arrays have been set.");
        size_t nstates = m_nk.shape(0);
        size_t N = m_nk.shape(1);
        std::vector<complex_type> alpha(N);

        for(size_t i = 0; i < N; ++i)
        {
            real_type sigma = 1.0/std::sqrt(2.0*(1-std::exp(-beta*m_wk[i])));
            std::normal_distribution<T> dist(0, sigma);

            alpha[i] = complex_type(dist(rng), dist(rng));
            std::cerr << alpha[i] << std::endl;
        }


        psi0.fill_zeros();


        linalg::matrix<complex_type> temp(nstates, nstates);
        form_displacement_operator(temp, alpha);

        linalg::vector<complex_type> hpsi0(nstates);

        for(size_t i=0; i<nstates; ++i)
        {
            hpsi0(i) = temp(i, 0);
        }
        complex_type dot_prod = linalg::dot_product(linalg::conj(hpsi0), hpsi0);
        
        hpsi0/=std::sqrt(std::real(dot_prod));
        std::cerr << hpsi0 << std::endl;
        psi0 = hpsi0;
    }

protected:
    size_t get_nstates_in_cut(size_t mode_ind, T ecut, size_t nmax)
    {
        T wi = std::abs(m_wk[mode_ind]);
        std::cerr << wi << std::endl;
        size_t nper = nmax < static_cast<size_t>(ecut/wi) ? nmax : static_cast<size_t>(ecut/wi);
        size_t res = 1;
        for(size_t i=0; i < m_max_states.size(); ++i){m_max_states[i] = nper;   res*=nper;}
        return res;
    }
    


    size_t populate_state_vector(linalg::matrix<size_t>& nk, linalg::vector<size_t>& nt, size_t curr_ind, size_t mode_ind, T ecut, size_t nmax)
    {
        T wi = std::abs(m_wk[mode_ind]);
        std::cerr << wi << std::endl;
        size_t nper = m_max_states[0];
        if(mode_ind < m_wk.size())
        {
            for(size_t i=0; i<nper; ++i)
            {   
                nt[mode_ind] = i;
                curr_ind = populate_state_vector(nk, nt, curr_ind, mode_ind+1, ecut, nmax);
            }
        }
        else{nk[curr_ind] = nt;  ++curr_ind;}
        return curr_ind;
    }


};


template <typename T>
bool partition_modes(const std::vector<T>& wk, T ecut, T wc, std::vector<size_t>& partition, size_t nmax, size_t nmax4, size_t nmax2, size_t nmax3)
{
    size_t N = wk.size();
    ASSERT(N != 0, "Failed to partition modes.  We cannot have zero modes.");


    for(size_t i = 0; i+1 < N; ++i)
    {
        //std::cerr << wk[i] << " " << wk[i+1] << std::endl;
        ASSERT(std::abs(wk[i+1]) >= std::abs(wk[i]), "Failed to partition modes.  The frequency array must be sorted from smallest to largest.");
    }

    std::list<size_t> partitions;
    harmonic_mode_combination<T> mode_combination;

    size_t nremaining = N;
    size_t count = 0;

    size_t npartitions = 0;
    while(nremaining != 0)
    {
        size_t nmaxt = wk[count] > wc ? nmax3 : nmax2;
        size_t nmaxp = wk[count] > wc ? nmax4 : nmax;

        bool find_n_to_partition = true;
        size_t ntoadd = 1;
        size_t nt = 0;

        while(find_n_to_partition)
        {
            mode_combination.set_wk(wk.begin()+count, wk.begin()+count + ntoadd); 
            nt = mode_combination.compute_nstates(ecut, nmaxt);
            if(nt > nmaxp)
            {
                if(ntoadd == 1)
                {
                    find_n_to_partition = false;
                }
                else
                {
                    --ntoadd;   
                    find_n_to_partition = false;
                }
            }
            else
            {
                if(nremaining > ntoadd)
                {
                    ++ntoadd;
                }
                else
                {
                    find_n_to_partition = false;
                }
            }
        }
        partitions.push_back(ntoadd);
        nremaining -= ntoadd;

        mode_combination.set_wk(wk.begin()+count, wk.begin()+count + ntoadd); 
        count += partitions.back();
        nt = mode_combination.compute_nstates(ecut, nmaxt);
        //std::cerr << npartitions << " " << ntoadd << " " << nt << std::endl;
        ++npartitions;
    }
    
    partition = std::vector<size_t>({partitions.begin(), partitions.end()});
    
    return true;
}   
#endif

