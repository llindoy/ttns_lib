#ifndef HTTENSOR_OPERATOR_COMPRESSION_HPP
#define HTTENSOR_OPERATOR_COMPRESSION_HPP

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
class minimal_index_array
{
public:
    static_assert(std::is_integral<size_type>::value, "Cannot create minimal array for non-integral type.");
public:
    minimal_index_array() : m_rindex(0), m_lindex(0), m_val(0){}
    minimal_index_array(const std::vector<size_type>& r, size_type maxr) 
    try : m_maxr(maxr), m_rindex(0), m_lindex(0), m_val(0)
    {
        set(r);
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct minimal_index_array object.");
    }
    minimal_index_array(const minimal_index_array& o) = default;
    minimal_index_array(minimal_index_array&& o) = default;

    minimal_index_array& operator=(const minimal_index_array& o) = default;
    minimal_index_array& operator=(minimal_index_array&& o) = default;

    void set(const std::vector<size_type>& l)
    {
        if(l.size() > m_maxr/2)
        {
            m_store_inverse = true; 
            m_r.resize(m_maxr - l.size());
            invert_vector(l, m_r, m_maxr);
        }
        else
        {
            m_store_inverse = false;
            m_r = l;
        }
    }

    void set(const std::vector<size_type>& l, size_type maxr)
    {
        m_maxr = maxr;
        set(l);
    }

    std::vector<size_type> get() const
    {
        if(m_store_inverse)
        {
            std::vector<size_type> ret(m_maxr - m_r.size());
            invert_vector(m_r, ret, m_maxr);
            return ret;
        }
        else
        {
            return m_r;
        }
    }

    std::vector<size_type>& get(std::vector<size_type>& ret) const
    {
        if(m_store_inverse)
        {
            ret.resize(m_maxr - m_r.size());
            invert_vector(m_r, ret, m_maxr);
            return ret;
        }
        else
        {
            ret = m_r;
            return ret;
        }
    }

    size_type size() const
    {
        if(m_store_inverse){return m_maxr - m_r.size();}
        else{return m_r.size();}
    }

    //this resets the indices so that we are setup so that rval is the first element of the r array.
    void reset_index()
    {
        m_rindex = 0;
        m_lindex = 0;
        m_val = size_type(0);
        this->operator++();
    }

    minimal_index_array& operator++()
    {
        if(!m_store_inverse)
        {
            if(m_rindex < m_r.size())
            {
                m_val = m_r[m_rindex];
            }
            ++m_rindex;
        }
        else
        {
            if(m_rindex < m_r.size())
            {
                bool continue_loop = m_r[m_rindex] == m_lindex;
                while(continue_loop)
                {
                    ++m_rindex;
                    ++m_lindex;
                    if(m_rindex < m_r.size()){continue_loop = m_r[m_rindex] == m_lindex;}
                    else{continue_loop = false;}
                }
                m_val = m_lindex;
                ++m_lindex;
            }
            else
            {
                m_val = m_lindex;
                ++m_lindex;
            }
        }
        return *this;
    }

    bool at_end() const
    {
        if(!m_store_inverse){return m_rindex > m_r.size();}
        else{return m_lindex > m_maxr;}
    }       
    
    const size_type& val() const{m_val;}

    bool contains(size_type v) const
    {
        if(!m_store_inverse){return std::find(m_r.begin(), m_r.end(), v) != m_r.end();}
        else{return std::find(m_r.begin(), m_r.end(), v) == m_r.end();}
    }


    void clear()
    {
        m_r.clear();
        m_rindex = 0;
        m_lindex = 0;
        m_val = size_type(0);
    }
protected:
    std::vector<size_type> m_r;                                   
    bool m_store_inverse;
    size_type m_maxr;
    size_type m_rindex;
    size_type m_lindex;
    size_type m_val;

public:
    static void invert_vector(const std::vector<size_type>& in, std::vector<size_type>& out, size_type maxr)
    {
        size_type lcount = 0;
        size_type rcount = 0;
        for(size_type i = 0; i < maxr; ++i)
        {
            if(lcount < in.size())
            {
                if(in[lcount] != i)
                {
                    out[rcount] = i;
                    ++rcount;
                }
                else
                {
                    ++lcount;
                }
            }
            else
            {
                out[rcount] = i;
                ++rcount;
            }
        }
    }

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void serialize(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::make_nvp("r", m_r)), "Failed to serialise minimal_index_array object.  Error when serialising the r array.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("store_inverse", m_store_inverse)), "Failed to serialise minimal_index_array object.  Error when serialising the store_inverse variable.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("maxr", m_maxr)), "Failed to serialise minimal_index_array object.  Error when serialising the maxr.");
    }
#endif
};

//structures for indexing the arrays
template <typename size_type>
class operator_index
{
public:
    operator_index() {}
    operator_index(bool isid, const std::vector<size_type>& r, size_type maxr) 
    try : m_is_identity(isid), m_r(r, maxr){}
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct operator_index object.");
    }
    operator_index(const operator_index& o) = default;
    operator_index(operator_index&& o) = default;

    operator_index& operator=(const operator_index& o) = default;
    operator_index& operator=(operator_index&& o) = default;
    
    bool is_identity() const{return m_is_identity;}

    const minimal_index_array<size_type>& r() const{return m_r;}
    minimal_index_array<size_type>& r(){return m_r;}

    bool contains_index(size_type r) const{return m_r.contains(r);}
protected:
    bool m_is_identity;
    minimal_index_array<size_type> m_r;
};


template <typename T, typename B> 
class operator_compression
{
public:
    using value_type = T;
    using backend_type = B;
    using size_type = typename backend_type::size_type;
    using operator_type = sop_operator<T, B>;
    using data_type = std::list<operator_index<size_type>>;
    using index_vector = std::vector<size_type>;

    using tree_type = tree<data_type>;
    using node_type = typename tree_type::node_type;

    using iterator_type = typename data_type::const_iterator;
protected:
    template <typename IterArr>
    static void advance_iterators(IterArr& ciiterators, const IterArr& ciiterators_begin, const IterArr& ciiterators_end, size_type uind, std::vector<bool>& requires_update, index_vector& hindex, std::vector<index_vector>& union_temporary)
    {
        ++ciiterators[uind];        
        ++hindex[uind];
        requires_update[uind] = true;
        
        //zero any additional iterators
        for(size_type i=uind+1; i < requires_update.size(); ++i)
        {
            ciiterators[i] = ciiterators_begin[i];        
            hindex[i] = 0;
            requires_update[i] = true;
            union_temporary[i].clear();
        }
        
        
        while(ciiterators[uind] == ciiterators_end[uind])
        {
            if(uind != 0)
            {
                ciiterators[uind] = ciiterators_begin[uind];
                hindex[uind] = 0;
                union_temporary[uind].clear();
                --uind;
        
                ++ciiterators[uind];
                ++hindex[uind];
                requires_update[uind] = true;
            }
            else
            {
                ciiterators[uind] = ciiterators_end[uind];
                return;
            }
        }
    }


    template <typename IterArr>
    static void process_operator_iterators(IterArr& ciiterators, const IterArr& ciiterators_begin, const IterArr& ciiterators_end, size_type maxterms, data_type& res, bool add_one_bodies = true)
    {
        size_type nch = ciiterators.size();
        index_vector hindex(nch); std::fill(hindex.begin(), hindex.end(), 0);

        std::vector<index_vector> union_temporary(nch);
        for(size_type i=0; i < nch; ++i){union_temporary.reserve(maxterms);}

        std::vector<index_vector> temp_vals(nch);
        for(size_type i=0; i < nch; ++i){temp_vals.reserve(maxterms);}

        std::vector<bool> requires_update(nch); std::fill(requires_update.begin(), requires_update.end(), true);        
        
        while(ciiterators[0] != ciiterators_end[0])
        {
            bool all_identity = ciiterators[0]->is_identity();

            if(requires_update[0])
            {
                ciiterators[0]->r().get(temp_vals[0]);
                union_temporary[0] = temp_vals[0];
                requires_update[0] = false;
            }
            
            bool keep_running = true;
            for(size_type i=1; (i < nch) && keep_running; ++i)
            {
                all_identity = all_identity && ciiterators[i]->is_identity();
                if(requires_update[i])
                {
                    union_temporary[i].clear();
                    ciiterators[i]->r().get(temp_vals[i]);
                    if(union_temporary[i-1].size() > 1 && temp_vals[i].size() > 1)
                    {
                        std::set_intersection(temp_vals[i].begin(), temp_vals[i].end(), union_temporary[i-1].begin(), union_temporary[i-1].end(), std::back_inserter(union_temporary[i]));
                    }
                    else if(union_temporary[i-1].size() == 1 && temp_vals[i].size() == 1 && add_one_bodies)
                    {
                        if(union_temporary[i-1].back() == temp_vals[i].back()){union_temporary[i].push_back(union_temporary[i-1].back());}
                    }
                    else if(union_temporary[i-1].size() == 1 && add_one_bodies)
                    {
                        if(std::binary_search(temp_vals[i].begin(), temp_vals[i].end(), union_temporary[i-1].back())){union_temporary[i].push_back(union_temporary[i-1].back());}
                    }
                    else if(temp_vals[i].size() == 1 && add_one_bodies)
                    {
                        if(std::binary_search(union_temporary[i-1].begin(), union_temporary[i-1].end(), temp_vals[i].back())){union_temporary[i].push_back(temp_vals[i].back());}
                    }
                    requires_update[i] = false;
                }
                if(union_temporary[i].size() == 0)
                {
                    keep_running = false;
                    advance_iterators(ciiterators, ciiterators_begin, ciiterators_end, i, requires_update, hindex, union_temporary);
                } 
            }
        
            size_type uind = nch-1;
        
            if(keep_running)
            {
                if(add_one_bodies || union_temporary.back().size() > 1)
                {
                    res.push_back(operator_index<size_type>(all_identity, {union_temporary.back().begin(), union_temporary.back().end()}, maxterms));
                    if(res.size() == maxterms){return;}
                }
                advance_iterators(ciiterators, ciiterators_begin, ciiterators_end, uind, requires_update, hindex, union_temporary);
            }
        }
    }

public:
    static void get_spf_node_info(const operator_type& ham, node_type& generic)
    {
        //if we are treating the leaf case then we just push the ham r-index info for this node into this object case 
        if(generic.is_leaf())
        {
            size_type nu = generic.leaf_index();
            data_type p;
            for(const auto& ti : ham.operators(nu))
            {
                if(ti.r().size() > 1)
                {
                    std::vector<size_type> rt(ti.r());
                    p.push_back(operator_index<size_type>(ti.is_identity(), rt, ham.nterms()));
                }
            }
            //now we get the identity terms
            std::vector<size_type> ridentity(ham.ridentity(nu));
            if(ridentity.size() > 1)
            {
                p.push_back(operator_index<size_type>(true, ridentity, ham.nterms()));
            }

            generic().resize(p.size());   std::copy(p.begin(), p.end(), generic().begin());
        }
        else
        {
            size_type nch = generic.size();

            bool m_compute_spf_info = true;
            for(size_type i=0; i < nch; ++i)
            {
                if(generic[i]().size() == 0){m_compute_spf_info = false;}
            }

            if(m_compute_spf_info)
            {
                std::vector<iterator_type> ciiterators;         ciiterators.reserve(nch);
                std::vector<iterator_type> ciiterators_begin;   ciiterators_begin.reserve(nch);
                std::vector<iterator_type> ciiterators_end;     ciiterators_end.reserve(nch);

                for(size_type i=0; i < nch; ++i)
                {
                    ciiterators.push_back(generic[i]().cbegin());
                    ciiterators_begin.push_back(generic[i]().cbegin());
                    ciiterators_end.push_back(generic[i]().cend());
                }
                data_type p;
                process_operator_iterators(ciiterators, ciiterators_begin, ciiterators_end, ham.nterms(), p, false);
                generic().resize(p.size());   std::copy(p.begin(), p.end(), generic().begin());
            }
            else
            {
                generic().resize(0);
            }
        }
    }

    static void get_mf_node_info(const node_type& spf, size_type max_nterms, node_type& mf)
    {
        //if this is the root node, then the mean field is common for each node and it is he identity
        if(mf.is_root())
        {
            std::vector<size_type> v(max_nterms);
            for(size_type i=0; i < max_nterms; ++i){v[i] = i;}
            bool is_id = true;
            mf().push_back(operator_index<size_type>(is_id, v, max_nterms));
        }
        else
        {
            //if we aren't at the root we get the parent objects for the single particle operators and we do a similar intersection process to determine which modes this operator is associated with 
            const auto& mf_p = mf.parent();
            const auto& spf_p = spf.parent();

            size_type nch = spf_p.size();

            size_type nu = mf.child_id();

            bool m_compute_mf_info = true;
            if(mf_p().size() == 0){m_compute_mf_info = false;}
            for(size_type i=0; i < nch; ++i)
            {
                if(i != nu)
                {
                    if(spf_p[i]().size() == 0){m_compute_mf_info = false;}
                }
            }

            if(m_compute_mf_info)
            {
                std::vector<iterator_type> ciiterators;         ciiterators.reserve(nch);
                std::vector<iterator_type> ciiterators_begin;   ciiterators_begin.reserve(nch);
                std::vector<iterator_type> ciiterators_end;     ciiterators_end.reserve(nch);

                ciiterators.push_back(mf_p().cbegin());
                ciiterators_end.push_back(mf_p().cend());
                ciiterators_begin.push_back(mf_p().cbegin());
                for(size_type i=0; i < nch; ++i)
                {
                    if(i != nu)
                    {
                        ciiterators.push_back(spf_p[i]().cbegin());
                        ciiterators_end.push_back(spf_p[i]().cend());
                        ciiterators_begin.push_back(spf_p[i]().cbegin());
                    }
                }

                data_type p;
                process_operator_iterators(ciiterators, ciiterators_begin, ciiterators_end, max_nterms, p, false);
                mf().resize(p.size());   std::copy(p.begin(), p.end(), mf().begin());
                p.clear();
            }
            else
            {
                mf().resize(0);
            }
        }
    }
};  //class operator_compression


}   //namespace ttns

#endif  //HTTENSOR_OPERATOR_COMPRESSION_HPP

