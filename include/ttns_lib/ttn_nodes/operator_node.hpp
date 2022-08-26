#ifndef HTTENSOR_OPERATOR_NODE_DATA_HPP
#define HTTENSOR_OPERATOR_NODE_DATA_HPP

#include <linalg/linalg.hpp>

#include <memory>
#include <list>
#include <vector>
#include <array>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <list>
#include <tuple>
#include <memory>
#include <utility>
#include <initializer_list>
#include <type_traits>

namespace ttns
{

template <typename size_type>
class mf_index
{
public:
    using container_type = std::vector<std::array<size_type, 2> >;

public:
    mf_index(){}
    mf_index(size_type s)
    {
        CALL_AND_HANDLE(resize(s), "Failed to construct mf_index object.");
    }

    mf_index(const mf_index& o) = default;
    mf_index(mf_index&& o) = default;
    mf_index(size_type parentindex, const container_type& v) : m_parent_index(parentindex), m_sibling_indices(v) {}
    mf_index(size_type parentindex, container_type&& v) : m_parent_index(parentindex), m_sibling_indices(std::move(v)) {}

    mf_index& operator=(const mf_index& o) = default;
    mf_index& operator=(mf_index&& o) = default;

    void clear()
    {
        m_parent_index = 0;
        m_sibling_indices.clear();
    }
    
    void resize(size_type size)
    {
        CALL_AND_HANDLE(m_sibling_indices.resize(size), "Failed to resize sibling indices.");
    }

    const size_type& parent_index() const{return m_parent_index;}
    size_type& parent_index(){return m_parent_index;}

    const std::array<size_type, 2> & sibling_index(size_type i) const{return m_sibling_indices[i];}
    std::array<size_type, 2> & sibling_index(size_type i){return m_sibling_indices[i];}

    const container_type& sibling_indices() const{return m_sibling_indices;}
#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void serialize(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::make_nvp("parent_index", m_parent_index)), "Failed to serialise mf_index object.  Error when serialising the parent_index.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("sibling_indices", m_sibling_indices)), "Failed to serialise mf_index object.  Error when serialising the sibling_indices.");
    }
#endif
protected:
    size_type m_parent_index;
    container_type m_sibling_indices;
};

template <typename T, typename B> class operator_node_data;

template <typename T, typename B> 
class operator_term 
{
public:
    using backend_type = B;
    using size_type = typename backend_type::size_type;
    using real_type = typename tmp::get_real_type<T>::type; 
    using hnode = httensor_node<T, B>;
    using triad = std::vector<linalg::matrix<T, B>>;

    using tree_type = tree<operator_node_data<T, B>>;
    using node_type = typename tree_type::node_type;

    using accum_coeff_type = std::vector<T>;

    using spf_index_type = std::vector<std::vector<std::array<size_type, 2>>>;
    using mf_index_type = std::vector<mf_index<size_type>>;

    template <typename Y, typename V> friend class operator_container;
public:
    operator_term() : m_is_identity_spf(false), m_is_identity_mf(false) {}
    operator_term(const operator_term& o) = default;
    operator_term(operator_term&& o) = default;
    operator_term& operator=(const operator_term& o) = default;
    operator_term& operator=(operator_term&& o) = default;

    template <typename be> 
    typename std::enable_if<not std::is_same<be, backend_type>::value, operator_term&>::type operator=(const operator_term<T, be> & o) 
    {
        try
        {
            CALL_AND_HANDLE(m_spf = o.spf(), "Failed to copy spf matrix.");
            CALL_AND_HANDLE(m_mf = o.mf(), "Failed to copy mf matrix.");
            
            CALL_AND_HANDLE(m_spf_index = o.spf_index(), "Failed to copy spf matrix.");
            CALL_AND_HANDLE(m_mf_index = o.mf_index(), "Failed to copy mf matrix.");

            m_accum_coeff = o.accum_coeff();
            m_coeff = o.coeff();
            m_is_identity_spf = o.is_identity_spf();
            m_is_identity_mf = o.is_identity_mf();
            return *this;
        }
        catch(const std::exception& ex){std::cerr << ex.what() << std::endl;    RAISE_EXCEPTION("Failed to copy assign operator term.");}
    }

    void resize_indexing(size_type nspfterms, size_type nmfterms)
    {
        m_spf_index.resize(nspfterms);
        m_mf_index.resize(nmfterms);  
    }
    void resize_indexing(const std::vector<size_type>& spfsize, const std::vector<size_type>& mfsize)
    {
        m_spf_index.resize(spfsize.size());   for(size_type i=0; i<spfsize.size(); ++i){m_spf_index.resize(spfsize[i]);}
        m_mf_index.resize(mfsize.size());   for(size_type i=0; i<mfsize.size(); ++i){m_mf_index[i].resize(mfsize[i]);}
    }

    void reallocate_matrices(size_type capacity)
    {
        if(!m_is_identity_spf)
        {
            CALL_AND_HANDLE(m_spf.reallocate(capacity), "Failed to resize spf matrix.");
        }
        if(!m_is_identity_mf)
        {
            CALL_AND_HANDLE(m_mf.reallocate(capacity), "Failed to resize spf matrix.");
        }
    }

    void resize_matrices(size_type n, size_type m)
    {
        m_matsize[0] = n;   m_matsize[1] = m;
        if(!m_is_identity_spf)
        {
            CALL_AND_HANDLE(m_spf.resize(n, m), "Failed to resize spf matrix.");
        }
        if(!m_is_identity_mf)
        {
            CALL_AND_HANDLE(m_mf.resize(n, m), "Failed to resize spf matrix.");
        }
    }

    const size_type& matrix_size(size_type i) const
    {
        ASSERT(i < 2, "Index out of bounds.");
        return m_matsize[i];
    }

    void clear() 
    {
        m_coeff = T(0);
        CALL_AND_HANDLE(m_accum_coeff.clear(), "Failed to clear the coefficient array.");
        CALL_AND_HANDLE(m_mf.clear(), "Failed to clear mf matrix object.");
        CALL_AND_HANDLE(m_spf.clear(), "Failed to clear spf matrix object.");
        CALL_AND_HANDLE(m_spf_index.clear(), "Failed to clear spf object.");
        CALL_AND_HANDLE(m_mf_index.clear(), "Failed to clear mf object.");
    }

    const linalg::matrix<T, B>& spf() const{return m_spf;}
    linalg::matrix<T, B>& spf(){return m_spf;}

    const linalg::matrix<T, B>& mf() const{return m_mf;}
    linalg::matrix<T, B>& mf(){return m_mf;}

    bool is_identity_spf() const{return m_is_identity_spf;}
    bool is_identity_mf() const{return m_is_identity_mf;}

    const accum_coeff_type& accum_coeff() const{return m_accum_coeff;}
    const T& accum_coeff(size_type i) const{return m_accum_coeff[i];}

    const T& coeff() const{return m_coeff;}

    const spf_index_type& spf_indexing() const{return m_spf_index;}
    const mf_index_type& mf_indexing() const{return m_mf_index;}
    
    size_type nspf_terms() const{return m_spf_index.size();}
    size_type nmf_terms() const{return m_mf_index.size();}

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void serialize(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::make_nvp("spf", m_spf)), "Failed to serialise operator term object.  Error when serialising the spf matrix.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("mf", m_mf)), "Failed to serialise operator term object.  Error when serialising the mf matrix.");

        CALL_AND_HANDLE(ar(cereal::make_nvp("is_identity_spf", m_is_identity_spf)), "Failed to serialise operator term object.  Error when serialising whether the spf matrix is the identity.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("is_identity_mf", m_is_identity_mf)), "Failed to serialise operator term object.  Error when serialising whether the mf matrix is the identity.");

        CALL_AND_HANDLE(ar(cereal::make_nvp("coeff", m_coeff)), "Failed to serialise operator term object.  Error when serialising the coefficient.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("accum_coeff", m_accum_coeff)), "Failed to serialise operator term object.  Error when serialising the coefficient.");

        CALL_AND_HANDLE(ar(cereal::make_nvp("spf_index", m_spf_index)), "Failed to serialise operator term object.  Error when serialising the spf indexing info.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("mf_index", m_mf_index)), "Failed to serialise operator term object.  Error when serialising the mf indexing info.");
    }
#endif

protected:
    linalg::matrix<T, B> m_spf;
    linalg::matrix<T, B> m_mf; 

    std::array<size_type, 2> m_matsize;

    spf_index_type m_spf_index;
    mf_index_type m_mf_index;

    bool m_is_identity_spf;
    bool m_is_identity_mf;

    accum_coeff_type m_accum_coeff;
    T m_coeff;
};

template <typename T, typename B>
class operator_node_data
{
public:
    using backend_type = B;
    using size_type = typename backend_type::size_type;
    using real_type = typename tmp::get_real_type<T>::type; 

    template <typename Y, typename V> friend class operator_container;
public:
    operator_node_data() : m_ncommon_spf(0), m_ncommon_mf(0), m_nstandard(0), m_skip1(0), m_skip2(0), m_use_rank_four_contraction(false){}
    operator_node_data(size_type commonspf, size_type commonmf, size_type standard) : m_use_rank_four_contraction(false)
    {
        CALL_AND_HANDLE(resize(commonspf, commonmf, standard), "Failed to resize operator node data object.");
    }
    operator_node_data(const operator_node_data& o) = default;
    operator_node_data(operator_node_data&& o) = default;
    operator_node_data& operator=(const operator_node_data& o) = default;
    operator_node_data& operator=(operator_node_data&& o) = default;

    template <typename be> 
    typename std::enable_if<not std::is_same<be, backend_type>::value, operator_node_data&>::type operator=(const operator_node_data<T, be> & o) 
    {
        try
        {
            m_term.resize(o.nterms());

            for(size_type i = 0; i < o.nterms(); ++i){m_term[i] = o.terms(i);}
            return *this;
        }
        catch(const std::exception& ex){std::cerr << ex.what() << std::endl;    RAISE_EXCEPTION("Failed to copy assign operator term.");}
    }

    ~operator_node_data(){}

    void resize_matrices(size_type n, size_type m)
    {
        for(size_type i=0; i < m_term.size(); ++i)
        {
            CALL_AND_HANDLE(m_term[i].resize_matrices(n, m), "Failed to setup matrices for operator node.");
        }
    }

    const size_type& matrix_size(size_type ind) const
    {
        ASSERT(m_term.size() > 0, "Index out of bounds.");
        CALL_AND_RETHROW(return m_term[0].matrix_size(ind));
    }

    void reallocate_matrices(size_type capacity)
    {
        for(size_type i=0; i < m_term.size(); ++i)
        {
            CALL_AND_HANDLE(m_term[i].reallocate_matrices(capacity), "Failed to setup matrices for operator node.");
        }
    }
    
    void resize(size_type commonspf, size_type commonmf, size_type standard)
    {
        m_ncommon_spf = commonspf;
        m_ncommon_mf = commonmf;
        m_nstandard = standard;
        m_skip1 = m_ncommon_spf;
        m_skip2 = m_ncommon_spf + m_ncommon_mf;
        size_type nterms = commonspf + commonmf + standard;
        CALL_AND_HANDLE(m_term.resize(nterms), "Failed to resize term array.");
    }

    void clear() 
    {
        try
        {
            for(size_type i = 0; i < m_term.size(); ++i){m_term[i].clear();}
            m_term.clear();
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear operator node object.");
        }
    }

    size_type nterms() const{return m_term.size();}

    const operator_term<T, B>& term(size_type i) const
    {
        ASSERT(i  < m_term.size(), "Index out of bounds.");
        return m_term[i];
    }

    operator_term<T, B>& term(size_type i)
    {
        ASSERT(i < m_term.size(), "Index out of bounds.");
        return m_term[i];
    }

    const operator_term<T, B>& operator[](size_type i) const {return m_term[i];}
    operator_term<T, B>& operator[](size_type i){return m_term[i];}

    const operator_term<T, B>& common_spf_term(size_type i) const{ASSERT(i  < m_ncommon_spf, "Index out of bounds.");   return m_term[i];}
    operator_term<T, B>& common_spf_term(size_type i){ASSERT(i  < m_ncommon_spf, "Index out of bounds.");   return m_term[i];}

    const operator_term<T, B>& common_mf_term(size_type i) const{ASSERT(i  < m_ncommon_mf, "Index out of bounds.");   return m_term[i+m_skip1];}
    operator_term<T, B>& common_mf_term(size_type i){ASSERT(i  < m_ncommon_mf, "Index out of bounds.");   return m_term[i+m_skip1];}

    const operator_term<T, B>& standard_term(size_type i) const{ASSERT(i  < m_nstandard, "Index out of bounds.");   return m_term[i+m_skip2];}
    operator_term<T, B>& standard_term(size_type i){ASSERT(i  < m_nstandard, "Index out of bounds.");   return m_term[i+m_skip2];}

    size_type ncommon_spf() const{return m_ncommon_spf;}
    size_type ncommon_mf() const{return m_ncommon_mf;}
    size_type nstandard() const{return m_nstandard;}

    bool use_rank_four_contraction() const{return m_use_rank_four_contraction;}
#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void serialize(archive& ar)
    {       
        CALL_AND_HANDLE(ar(cereal::make_nvp("terms", m_term)), "Failed to serialise operator node object.  Error when serialising the terms.");
    }
#endif

protected:
    std::vector<operator_term<T, B>> m_term;
    size_type m_ncommon_spf;
    size_type m_ncommon_mf;
    size_type m_nstandard;
    size_type m_skip1;
    size_type m_skip2;
    bool m_use_rank_four_contraction;
};  //operator_node_data

namespace node_data_traits
{
    //clear traits for the operator node data object
    template <typename T, typename backend>
    struct clear_traits<operator_node_data<T, backend> > 
    {
        void operator()(operator_node_data<T, backend>& t){CALL_AND_RETHROW(t.clear());}
    };
}   //namespace node_data_traits
}   //namespace ttns

#endif  //HTTENSOR_OPERATOR_NODE_DATA_HPP

