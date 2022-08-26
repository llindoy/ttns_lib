#ifndef HTTENSOR_SP_OPERATOR_CONTAINER_HPP
#define HTTENSOR_SP_OPERATOR_CONTAINER_HPP

#include <linalg/linalg.hpp>

#include <memory>
#include <list>
#include <vector>
#include <algorithm>

#include <linalg/linalg.hpp>
#include "primitive_operator.hpp"
#include "direct_product_operator.hpp"
#include "matrix_operators.hpp"
#include "liouville_space_operator.hpp"

#ifdef CEREAL_LIBRARY_FOUND
#include <cereal/types/vector.hpp>
#endif

namespace ttns
{

template <typename T, typename backend>
class mode_op_traits
{
public:
    using op_type = ops::primitive<T, backend>;
    using pointer_type = std::shared_ptr<op_type>;

    using backend_type = backend;
    using value_type = T;

    using self_type = mode_op_traits<T, backend>;
    using size_type = typename backend::size_type;
    using container_type = std::vector<size_type>;
    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;
    using reverse_iterator = typename container_type::reverse_iterator;
    using const_reverse_iterator = typename container_type::const_reverse_iterator;

public:
    mode_op_traits(){}
    mode_op_traits(size_type nterms)
    {
        try {m_r.reserve(nterms);}  
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct mode_op_traits object.");
        }
    }
    mode_op_traits(const container_type& r) 
    try : m_r(r) {CALL_AND_HANDLE(std::sort(m_r.begin(), m_r.end()), "Failed to sort r array.");}
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct mode_op_traits object.");
    }
        
    mode_op_traits(container_type&& r) 
    try : m_r(std::move(r)) {CALL_AND_HANDLE(std::sort(m_r.begin(), m_r.end()), "Failed to sort r array.");}
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct mode_op_traits object.");
    }

    mode_op_traits(const mode_op_traits& o)
    {
        try
        {
            m_op = std::make_shared<ops::primitive<T, backend>>(*o.m_op);
            CALL_AND_HANDLE(m_r = o.m_r, "Failed to copy indices array.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to copy construct mode_op_traits object.");
        }
    }

    mode_op_traits(mode_op_traits&& o) = default;

    mode_op_traits& operator=(const mode_op_traits& o)
    {
        if(&o != this)
        {
            try
            {
                m_op = std::make_shared<ops::primitive<T, backend>>(*o.m_op);
                CALL_AND_HANDLE(m_r = o.m_r, "Failed to copy indices array.");
            }
            catch(const std::exception& ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to copy construct mode_op_traits object.");
            }
        }
    }

    mode_op_traits& operator=(mode_op_traits&& o) = default;

    const container_type& r() const{return m_r;}
    size_type nterms() const{return m_r.size();}

    void append_indices(const container_type& _r)
    {
        try
        {
            for(auto& r : _r){m_r.push_back(r);}
            std::sort(m_r.begin(), m_r.end());
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to append indices.");
        }
    }

    iterator begin() {  return iterator(m_r.begin());  }
    iterator end() {  return iterator(m_r.end());  }
    const_iterator begin() const {  return const_iterator(m_r.begin());  }
    const_iterator end() const {  return const_iterator(m_r.end());  }

    reverse_iterator rbegin() {  return reverse_iterator(m_r.rbegin());  }
    reverse_iterator rend() {  return reverse_iterator(m_r.rend());  }
    const_reverse_iterator rbegin() const {  return const_reverse_iterator(m_r.rbegin());  }
    const_reverse_iterator rend() const {  return const_reverse_iterator(m_r.rend());  }

    void clear()
    {
        m_r.clear();
        m_op = nullptr;
    }

    template <typename mode_operator> 
    typename std::enable_if<std::is_base_of<op_type, mode_operator>::value, std::shared_ptr<ops::primitive<T, backend>>>::type bind(mode_operator&& op)
    {
        try
        {
            //if the container index associated with the current mode is non-zero then we need to unbind the current operator
            if(m_op != nullptr)
            {
                m_op = nullptr;
                m_op = std::make_shared<mode_operator>(std::move(op));
            }
            else
            {
                m_op = std::make_shared<mode_operator>(std::move(op));
            }
            return m_op;
        }
        catch(const std::exception& ex) 
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to bind operator to mode_op_traits object.");
        }
    }

    template <typename mode_operator> 
    typename std::enable_if<std::is_base_of<op_type, mode_operator>::value, std::shared_ptr<ops::primitive<T, backend>>>::type bind(const mode_operator& op)
    {
        try
        {
            //if the container index associated with the current mode is non-zero then we need to unbind the current operator
            if(m_op != nullptr)
            {
                m_op = nullptr;
                m_op = std::make_shared<mode_operator>(op);
            }
            else
            {
                m_op = std::make_shared<mode_operator>(op);
            }
            return m_op;
        }
        catch(const std::exception& ex) 
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to bind operator to mode_op_traits object.");
        }
    }

    template <typename A, typename B>
    void apply(const A& a, B& b) const
    {
        if(m_op != nullptr){CALL_AND_RETHROW(m_op->apply(a, b));}
        else{b = a;}
    }

    template <typename A, typename B, typename Rt>
    void apply(const A& a, B& b, Rt t, Rt dt) const
    {
        if(m_op != nullptr){CALL_AND_RETHROW(m_op->apply(a, b, t, dt));}
        else{b = a;}
    }   

    template <typename RT>
    void update(RT t, RT dt)
    {
        if(m_op != nullptr){CALL_AND_RETHROW(m_op->update(t, dt));}
    }

    bool is_identity() const
    {
        if(m_op != nullptr){return m_op->is_identity();}
        else{return true;}
    }

    size_type mode_dimension() const
    {
        if(m_op != nullptr){return m_op->size();}
        else{return 0;}
    }

    bool contains_index(size_type r) const{return (std::find(m_r.begin(), m_r.end(), r) != m_r.end());}
protected:
    container_type m_r;
    std::shared_ptr<ops::primitive<T, backend>> m_op;

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void serialize(archive& ar) 
    {
        CALL_AND_HANDLE(ar(cereal::make_nvp("r", m_r)), "Failed to serialise mode operator.  Failed to serialise r array.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("ops", m_op)), "Failed to serialise mode operator.  Failed to serialise operators.");
    }
#endif
};

//a generic sum of product operator object.
template <typename T, typename backend = linalg::blas_backend>
class sop_operator
{
public:
    using size_type = typename backend::size_type;
    using element_type = mode_op_traits<T, backend>;
    using op_type = ops::primitive<T, backend>;

    using element_container_type = typename element_type::container_type;

    using mode_terms_type = std::vector<element_type>;
    using container_type = std::vector<mode_terms_type>;
    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;
    using reverse_iterator = typename container_type::reverse_iterator;
    using const_reverse_iterator = typename container_type::const_reverse_iterator;

    sop_operator() : m_nterms(0){}
    sop_operator(size_type nterms, const std::vector<size_type>& mode_dimensions) 
    try : m_mode_operators(mode_dimensions.size()), m_nterms(nterms), m_mode_dimension(mode_dimensions), m_coeff(nterms)
    {
        std::fill(m_coeff.begin(), m_coeff.end(), T(1));
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct sum of product operator object.");
    }    
    sop_operator(size_type nterms, const std::vector<size_type>& mode_dimensions, const std::vector<size_type>& nterms_per_mode)
    try : m_mode_operators(mode_dimensions.size()), m_nterms(nterms), m_mode_dimension(mode_dimensions), m_coeff(nterms)
    {   
        for(size_type i = 0; i < mode_dimensions.size(); ++i)
        {
            m_mode_operators[i].reserve(nterms_per_mode[i]);
        }
        std::fill(m_coeff.begin(), m_coeff.end(), T(1));
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct sum of product operator object.");
    }

    sop_operator(const sop_operator& o) = default;
    sop_operator(sop_operator&& o) = default;

    sop_operator& operator=(const sop_operator& o) = default;
    sop_operator& operator=(sop_operator&& o) = default;

    void resize(size_type nterms, const std::vector<size_type>& mode_dimensions)
    {
        try
        {
            clear();
            m_mode_operators.resize(mode_dimensions.size());
            m_mode_dimension = mode_dimensions;
            m_coeff.resize(nterms); std::fill(m_coeff.begin(), m_coeff.end(), T(1));
            for(size_type i = 0; i < mode_dimensions.size(); ++i)
            m_nterms = nterms;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize sp hamiltonian object.");
        }
    }
   
    void resize(size_type nterms, const std::vector<size_type>& mode_dimensions, const std::vector<size_type>& nterms_per_mode)
    {
        try
        {
            clear();
            m_mode_operators.resize(mode_dimensions.size());
            m_mode_dimension = mode_dimensions;
            m_coeff.resize(nterms); std::fill(m_coeff.begin(), m_coeff.end(), T(1));
            for(size_type i = 0; i < mode_dimensions.size(); ++i)
            {
                m_mode_operators[i].reserve(nterms_per_mode[i]);
            }
            m_nterms = nterms;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize sp hamiltonian object.");
        }
    }
   

    std::shared_ptr<ops::primitive<T, backend>> bind(const mode_op_traits<T, backend>& op, size_type nu) 
    {
        //first we check that non of the r-indices in this object have already been bound
        ASSERT(valid_rvals(op.r(), nu), "Unable to bind operator to sum of product operator.  At least one of the r indices specified has previously been bound for this mode.");

        //we add in the operator if it isn't 
        if(!op.is_identity())
        {
            ASSERT(op.size() == m_mode_dimension[nu], "Failed to bind operator.  It does not have the correct size.");
            CALL_AND_HANDLE(return m_mode_operators[nu].push_back(op), "Failed to push mode operator term to list.");
        }
        else
        {
            return nullptr;
        }
    }

    std::shared_ptr<ops::primitive<T, backend>> bind(mode_op_traits<T, backend>&& op, size_type nu) 
    {
        ASSERT(valid_rvals(op.r(), nu), "Unable to bind operator to sum of product operator.  At least one of the r indices specified has previously been bound for this mode.");

        if(!op.is_identity())
        {
            ASSERT(op.size() == m_mode_dimension[nu], "Failed to bind operator.  It does not have the correct size.");
            CALL_AND_HANDLE(return m_mode_operators[nu].push_back(std::move(op)), "Failed to push mode operator term to list.");
        }
        else
        {
            return nullptr;
        }
    }

    template <typename mode_operator> 
    typename std::enable_if<std::is_base_of<op_type, mode_operator>::value, std::shared_ptr<ops::primitive<T, backend>>>::type bind(const mode_operator& op, const element_container_type& r, size_type nu)
    {
        //first we check that non of the r-indices in this object have already been bound
        ASSERT(valid_rvals(r, nu), "Unable to bind operator to sum of product operator.  At least one of the r indices specified has previously been bound for this mode.");
        ASSERT(op.size() == m_mode_dimension[nu], "Failed to bind operator.  It does not have the correct size.");

        //we add in the operator if it isn't 
        if(!op.is_identity())
        {
            ASSERT(op.size() == m_mode_dimension[nu], "Failed to bind operator.  It does not have the correct size.");
            CALL_AND_HANDLE(m_mode_operators[nu].push_back(mode_op_traits<T, backend>(r)), "Failed to push mode operator term to list.");
            return m_mode_operators[nu].back().bind(op);
        }
        else
        {
            return nullptr;
        }
    } 
    
    template <typename mode_operator> 
    typename std::enable_if<std::is_base_of<op_type, mode_operator>::value, std::shared_ptr<ops::primitive<T, backend>>>::type bind(mode_operator&& op, const element_container_type& r, size_type nu)
    {
        //first we check that non of the r-indices in this object have already been bound
        ASSERT(valid_rvals(r, nu), "Unable to bind operator to sum of product operator.  At least one of the r indices specified has previously been bound for this mode.");

        //we add in the operator if it isn't 
        if(!op.is_identity())
        {
            ASSERT(op.size() == m_mode_dimension[nu], "Failed to bind operator.  It does not have the correct size.");
            CALL_AND_HANDLE(m_mode_operators[nu].push_back(mode_op_traits<T, backend>(r)), "Failed to push mode operator term to list.");
            return m_mode_operators[nu].back().bind(std::move(op));
        }
        else
        {
            return nullptr;
        }
    } 
    
    bool is_identity(size_type r, size_type nu) const
    {
        for(const auto& li : m_mode_operators[nu])
        {
            if(li.contains_index(r)){return false;}
        }
        return true;
    }

    size_type index(size_type r, size_type nu) const
    {
        size_type count = 0;
        for(const auto& li : m_mode_operators[nu])
        {
            if(li.contains_index(r)){return count;}
            ++count;
        }
        return count;
    }

    void clear()
    {
        m_mode_operators.clear();
        m_mode_dimension.clear();       
        m_coeff.clear();
        m_nterms = 0;
    }

    const mode_terms_type& operators(size_type nu) const
    {
        return m_mode_operators[nu];
    }

    const mode_terms_type& operator[](size_type nu) const
    {
        return m_mode_operators[nu];
    }

    const mode_terms_type& operator()(size_type nu) const
    {
        return m_mode_operators[nu];
    }

    const element_type& operator()(size_type nu, size_type k) const
    {
        return m_mode_operators[nu][k];
    }

    template <typename RT>
    void update(size_type nu, RT t, RT dt)
    {
        for(size_t term = 0; term < m_mode_operators[nu].size(); ++term)
        {
            m_mode_operators[nu][term].update(t, dt);
        }
    }

    //need to check that this is working correctly
    element_container_type ridentity(size_type nu) const
    {
        size_type nrbound = 0;
        std::list<size_type> bound;
        for(auto& combop : m_mode_operators[nu]){nrbound += combop.nterms();}
        
        element_container_type ret; ret.reserve(m_nterms - nrbound);
        for(size_type i=0; i < m_nterms; ++i)
        {
            bool insert_element = true;
            for(auto& combop : m_mode_operators[nu])
            {
                insert_element = insert_element && (std::find(combop.r().begin(), combop.r().end(), i) == combop.r().end());
            }
            if(insert_element){ret.push_back(i);}
        }
        return ret;
    }

    const std::vector<T>& coeff() const{return m_coeff;}
    const T& coeff(size_type r)const{return m_coeff[r];}
    T& coeff(size_type r){return m_coeff[r];}

    size_type nterms() const{return m_nterms;}
    size_type nterms(size_type nu) const{return m_mode_operators[nu].size();}
    size_type nmodes() const{return m_mode_operators.size();}

    iterator begin() {  return iterator(m_mode_operators.begin());  }
    iterator end() {  return iterator(m_mode_operators.end());  }
    const_iterator begin() const {  return const_iterator(m_mode_operators.begin());  }
    const_iterator end() const {  return const_iterator(m_mode_operators.end());  }

    reverse_iterator rbegin() {  return reverse_iterator(m_mode_operators.rbegin());  }
    reverse_iterator rend() {  return reverse_iterator(m_mode_operators.rend());  }
    const_reverse_iterator rbegin() const {  return const_reverse_iterator(m_mode_operators.rbegin());  }
    const_reverse_iterator rend() const {  return const_reverse_iterator(m_mode_operators.rend());  }

protected:
    //check whether any of the new r values are already bound for this mode.
    bool valid_rvals(const element_container_type& r, size_type nu)
    {
        for(auto& combop : m_mode_operators[nu])
        {
            for(const auto& ri : combop)
            {
                if(std::find(r.begin(), r.end(), ri) != r.end()){return false;}
            }
        }
        return true;
    }

protected:
    container_type m_mode_operators;
    size_type m_nterms;
    std::vector<size_type> m_mode_dimension;
    std::vector<T> m_coeff;

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void serialize(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::make_nvp("operators", m_mode_operators)), "Failed to serialise sum of product operator.  Failed to serialise array of product operators.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("coefficients", m_coeff)), "Failed to serialise sum of product operator.  Failed to serialise array of coefficients.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("nterms", m_nterms)), "Failed to serialise sum of product operator.  Failed to serialise the number of terms.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("mode_dimension", m_mode_dimension)), "Failed to serialise sum of product operator.  Failed to serialise the number of modes.");
    }
#endif
};  //class sop_operator


}   //namespace ttns

#endif  //HTTENSOR_SP_OPERATOR_CONTAINER_HPP

