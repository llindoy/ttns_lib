#ifndef HTTENSOR_OPERATOR_CONTAINER_HPP
#define HTTENSOR_OPERATOR_CONTAINER_HPP

#include "operator_compression.hpp"
#include "../operators/sop_operator.hpp"
#include "../ttn_nodes/operator_node.hpp"

namespace ttns
{
template <typename T, typename B>
class operator_container
{
public:
    using value_type = T;
    using real_type = typename tmp::get_real_type<T>::type; 
    using backend_type = B;
    using size_type = typename backend_type::size_type;

public:
    using compress = operator_compression<T, B>;
    using htnode = typename httensor<T, B>::node_type;
    using op_info_node = typename compress::node_type;

    using rindex_type = std::vector<minimal_index_array<size_type> >;
    using index_node = typename tree<rindex_type>::node_type;

    using node_data_type = operator_node_data<T, B>;
    using tree_type = tree<node_data_type>;
    using node_type = typename tree_type::node_type;

public:
    operator_container() {}
    operator_container(const httensor<T, B>& A, sop_operator<T, B>& op, bool use_capacity = false)
    {
        CALL_AND_HANDLE(resize(A, op, use_capacity), "Failed to construct operator_container object.");
    }
    operator_container(const operator_container& o) = default;
    operator_container(operator_container&& o) = default;

    operator_container& operator=(const operator_container& o) = default;
    operator_container& operator=(operator_container&& o) = default;

    void resize(const httensor<T, B>& A, const sop_operator<T, B>& op, bool use_capacity = false)
    {
        CALL_AND_RETHROW(clear());
        //allocate the indexing arrays that are used to determine which operator terms can be combined together.
        typename compress::tree_type spf_indices, mf_indices;

        CALL_AND_HANDLE(spf_indices.construct_topology(A), "Failed to construct topology of spf_indices.");
        CALL_AND_HANDLE(mf_indices.construct_topology(A), "Failed to construct topology of mf_indices.");
        CALL_AND_HANDLE(m_op.construct_topology(A), "Failed to construct topology of mf_indices.");

        //and build the spf indices from the operator object
        for(auto& n : reverse(spf_indices))
        {
            CALL_AND_HANDLE(compress::get_spf_node_info(op, n), "Failed to compute spf node info.");
        }
        std::cerr << "common spf indices initialised" << std::endl;

        //and build the mf indices from the spf indices;
        for(auto z1 : zip(spf_indices, mf_indices))
        {
            const auto& spf = std::get<0>(z1);   auto& mf = std::get<1>(z1);
            CALL_AND_HANDLE(compress::get_mf_node_info(spf, op.nterms(), mf), "Failed to compute mf node info.");
        }
        std::cerr << "common mf indices initialised" << std::endl;

        //now construct the operator indexing tree
        tree<rindex_type> indexes;
        CALL_AND_HANDLE(indexes.construct_topology(A), "Failed to construct operator tree topology.");

        //now we iterate up the trees and construct the spf portion of the indexing node objects. 
        size_type count = 0;
        for(auto z1 : rzip(spf_indices, mf_indices, indexes, m_op))
        {
            const auto& spf = std::get<0>(z1);  const auto& mf = std::get<1>(z1);   auto& n = std::get<2>(z1);  auto& res = std::get<3>(z1);
            CALL_AND_HANDLE(construct_spf_node_indices(spf, mf, op, n, res), "Fialed to setup the spf node indices.");
            ++count;
        }
        std::cerr << "spf initialised" << std::endl;

        //now we iterate down the tree and set up the mean field indices.
        for(auto z1 : rzip(A, indexes, m_op))
        {
            const auto& a = std::get<0>(z1);   auto& n = std::get<1>(z1);  auto& res = std::get<2>(z1);
            CALL_AND_HANDLE(construct_mf_node_indices(op, n, res), "Failed to setup the mf node indices.");
            if(use_capacity)
            {
                CALL_AND_HANDLE(res().reallocate_matrices(a().hrank(use_capacity)*a().hrank(use_capacity)), "Failed to setup matrices for the node.");
            }
            CALL_AND_HANDLE(res().resize_matrices(a().hrank(), a().hrank()), "Failed to setup matrices for the node.");
        }
        std::cerr << "mf initialised" << std::endl;
        indexes.clear();
    }

    void clear()
    {
        CALL_AND_HANDLE(m_op.clear(), "Failed to clear operator tree.");
    }

    tree_type& op(){return m_op;}
    const tree_type& op() const{return m_op;}

    T e() const
    {
        T ev(0);
        for(size_type i=0; i < m_op[0]().nterms(); ++i)
        {
            if(!m_op[0]()[i].is_identity_spf())
            {
                ev += m_op[0]()[i].spf()(0,0)*m_op[0]()[i].coeff();
            }
            else
            {
                ev += m_op[0]()[i].coeff();
            }
        }
        return ev;
    }
#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void serialize(archive& ar) 
    {
        CALL_AND_HANDLE(ar(cereal::make_nvp("op", m_op)), "Failed to serialise operator container.");
    }
#endif

protected:

    static size_type find_index(const rindex_type& m_r, size_type r)
    {
        for(size_type i = 0; i < m_r.size(); ++i){if(m_r[i].contains(r)){return i;}}
        RAISE_EXCEPTION("r index not found");
    }

    static void initialise_common_spf_upwards(const op_info_node& spf, const sop_operator<T, B>& op, index_node& n, node_type& res)
    {
        //get all of the spf indices constructed for the case where the single particle functions are common.
        //In this case the r values associated with the mode below are always larger than the rvalues associated with this mode
        //There is no chance of an overlap of r values

        size_type count = 0;
        std::vector<size_type> rtemp;   rtemp.reserve(op.nterms());
        //if this is the leaf then we just find the common indices and add it
        if(n.is_leaf())
        {
            for(const auto& spf_op : spf())
            {
                n()[count] = spf_op.r();    rtemp = spf_op.r().get();

                size_type nu = n.leaf_index();
                size_type cind = op.index(rtemp[0], nu);
                if(!spf_op.is_identity())
                {
                    res().common_spf_term(count).m_spf_index.resize(1);
                    res().common_spf_term(count).m_spf_index[0] = {{nu, cind}};
                }

                res().common_spf_term(count).m_is_identity_spf = spf_op.is_identity();
                res().common_spf_term(count).m_is_identity_mf = false;

                ++count;
            }
        }
        else
        {
            for(const auto& spf_op : spf())
            {
                n()[count] = spf_op.r();    rtemp = spf_op.r().get();

                size_type nu = 0;
                std::list<std::array<size_type, 2>> list;
                //now we iterate over the nodes 
                for(auto z1 : zip(n, res))
                {
                    const auto& rn = std::get<0>(z1);   auto& child = std::get<1>(z1);
                    //as the common_spf nodes below are always larger than those above.  We can simply search to find
                    //any single value in the node below.  
                    size_type crind = find_index(rn(), rtemp[0]);
                    if(!child()[crind].is_identity_spf()){list.push_back({{nu, crind}});}
                    ++nu;
                }
                if(list.size() != 0)
                {
                    res().common_spf_term(count).m_spf_index.resize(1);
                    res().common_spf_term(count).m_spf_index[0] = {list.begin(), list.end()};
                }
                res().common_spf_term(count).m_is_identity_spf = spf_op.is_identity();
                res().common_spf_term(count).m_is_identity_mf = false;
                ++count;
            }
        }
    }
    
    static void initialise_common_mf_upwards(const op_info_node& mf, const sop_operator<T, B>& op, std::vector<size_type>& r_remaining, index_node& n, node_type& res)
    {
        size_type count = 0;
        std::vector<size_type> rnot_added;  rnot_added.reserve(op.nterms());
        std::vector<size_type> rtemp;   rtemp.reserve(op.nterms());

        size_type skip = res().ncommon_spf();
        //group all of the spf operators that are part of the mean field operator common term.  These 
        //can all be accumulated.  If the spf operators are part of an spf_common operator then they are 
        //not added at this point as they are included within the spf_common term
        for(const auto& mf_op : mf())
        {
            rnot_added.clear();
            rtemp = mf_op.r().get();

            std::set_intersection(rtemp.begin(), rtemp.end(), r_remaining.begin(), r_remaining.end(), std::back_inserter(rnot_added));
            
            minimal_index_array<size_type> rmin(rnot_added, op.nterms());
            n()[count+skip] = rmin;

            std::list<std::vector<std::array<size_type, 2>>> spfop;
            std::vector<T> coeff;
            //if we are at the leaf node, we cannot have any common terms, and as such for each term there is only a single r value
            //we therefore only need to find the rindex of each independent term in rnot_added to determine this term
            if(n.is_leaf())
            {
                size_type nu = n.leaf_index();

                std::vector<std::array<size_type, 2> > temp(1);
                std::vector<std::array<size_type, 2> > empty;
                coeff.resize(rnot_added.size());
                for(size_type rind = 0; rind < rnot_added.size(); ++rind)
                {
                    coeff[rind] = op.coeff(rnot_added[rind]);
                    if(!op.is_identity(rnot_added[rind], nu))
                    {   
                        size_type cind = op.index(rnot_added[rind], nu);
                        temp[0] = {{nu, cind}};
                        spfop.push_back(temp);
                    }
                    //if the term is the identity operator then we just push back an empty vector. There is no index for it in the sop_operator object
                    else
                    {
                        spfop.push_back(empty);
                    }
                }
            }
            //for a branch node,  we need to compute set intersections with all children nodes to determine which spf values are combined together to evaluate this term.
            //This is necessary as it will allow us to compute the coefficient required to evaluate this term and avoids adding terms multiple times
            else
            {
                std::list<std::vector<size_type>> rres; 
                std::list<std::vector<size_type>> nids;

                using iterator_type = typename std::vector<minimal_index_array<size_type> >::const_iterator;

                size_type nch = n.size();
                std::vector<iterator_type> ciiterators;         ciiterators.reserve(nch);
                std::vector<iterator_type> ciiterators_begin;   ciiterators_begin.reserve(nch);
                std::vector<iterator_type> ciiterators_end;     ciiterators_end.reserve(nch);

                for(size_type i=0; i < nch; ++i)
                {
                    ciiterators.push_back(n[i]().cbegin());
                    ciiterators_begin.push_back(n[i]().cbegin());
                    ciiterators_end.push_back(n[i]().cend());
                }
                process_operator_iterators(ciiterators, ciiterators_begin, ciiterators_end, rnot_added, rres, nids);

                //contains the distinct terms we need to compute
                coeff.resize(rres.size());  std::fill(coeff.begin(), coeff.end(), T(0));
                size_type index = 0;

                //now that we have the child indices that have the correct r values we go about setting everything up.
                for(auto z : zip(rres, nids))
                {   
                    const auto& rinds = std::get<0>(z);  const auto& ni = std::get<1>(z);
                    
                    //set the coefficient array.  If there is only a single r term for this one, then the children are associated with a standard 
                    //term and we should store its coeff.  If there is more than one r term, then this contribution is a direct product of an accumulated
                    //term and a series of common terms.  The accumulated term already contains all of the coefficients and so we set the coefficient for
                    //this term equal to 1.
                    if(rinds.size() == 1){coeff[index] = op.coeff(rinds[0]);}
                    else{coeff[index] = T(1);}

                    //now we create the sparse representation of the indices array.
                    std::list<std::array<size_type, 2>> sparse_ni;
                    for(size_type i=0; i < ni.size(); ++i)
                    {
                        if(!res[i]()[ni[i]].is_identity_spf())
                        {
                            sparse_ni.push_back({{i, ni[i]}});
                        }
                    }
                    spfop.push_back({sparse_ni.begin(), sparse_ni.end()});
                    ++index;
                }
            }

            ASSERT(spfop.size() != 0, "The common_mf operator must have at least one non-identity spf contribution as otherwise it either has 1 function in total (and is therefore a standard term.  Or it has a common spf.");

            //now we add all of the information gathered to the node
            
            res().common_mf_term(count).m_accum_coeff = std::move(coeff);
            res().common_mf_term(count).m_coeff = T(1.0);
            res().common_mf_term(count).m_is_identity_spf = false;
            res().common_mf_term(count).m_is_identity_mf = mf_op.is_identity();
            res().common_mf_term(count).m_spf_index.resize(1);
            res().common_mf_term(count).m_spf_index = {spfop.begin(), spfop.end()};
            ++count;
        }
    }

    static void initialise_standard_upwards(const sop_operator<T, B>& op, std::vector<size_type>& r_remaining, index_node& n, node_type& res)
    {
        size_type count = 0;
        size_type skip = res().ncommon_spf() + res().ncommon_mf();
        for(size_type r : r_remaining)
        {
            std::vector<size_type> rmat(1); rmat[0] = r;
            minimal_index_array<size_type> rmin(rmat, op.nterms());
            n()[count+skip] = rmin;

            res()[count+skip].m_accum_coeff.resize(1);
            if(n.is_leaf())
            {
                size_type nu = n.leaf_index();
                if(!op.is_identity(r, nu))
                {   
                    size_type cind = op.index(r, nu);
                    res().standard_term(count).m_spf_index.resize(1);
                    res().standard_term(count).m_spf_index[0] = {{nu, cind}};
                    res().standard_term(count).m_is_identity_spf = false;
                }
                else
                {
                    res().standard_term(count).m_is_identity_spf = true;
                }
                res().standard_term(count).m_accum_coeff[0] = T(1.0);
                res().standard_term(count).m_coeff = op.coeff(r);
            }
            else
            {
                size_type nu = 0;
                std::list<std::array<size_type, 2>> list;
                //now we iterate over the nodes 
                for(auto z1 : zip(n, res))
                {
                    const auto& rn = std::get<0>(z1);   auto& child = std::get<1>(z1);
                    //as the common_spf nodes below are always larger than those above.  We can simply search to find
                    //any single value in the node below.  
                    size_type crind = find_index(rn(), r);
                    if(!child()[crind].is_identity_spf()){list.push_back({{nu, crind}});}
                    ++nu;
                }
                if(list.size() != 0)
                {
                    res().standard_term(count).m_spf_index.resize(1);
                    res().standard_term(count).m_spf_index[0] = {list.begin(), list.end()};
                    res().standard_term(count).m_is_identity_spf = false;
                }
                else
                {
                    res().standard_term(count).m_is_identity_spf = true;
                }
                res().standard_term(count).m_accum_coeff[0] = T(1.0);
                res().standard_term(count).m_coeff = op.coeff(r);
            }
            ++count;
        }
    }

    static void remove_common_terms(const op_info_node& spf, const op_info_node& mf, size_type nterms, std::vector<size_type>& r_remaining_spf, std::vector<size_type>& r_remaining)
    {
        std::vector<size_type> rtemp;   rtemp.reserve(nterms);
        size_type nremoved = 0;
        for(const auto& spf_op : spf())
        {
            rtemp.clear();  spf_op.r().get(rtemp);
            for(size_type ri : rtemp)
            {
                r_remaining[ri] = nterms; 
                ++nremoved;
            }   
        }
        {
            r_remaining_spf.resize(nterms - nremoved);
            size_type count = 0;
            for(size_type i = 0; i < nterms; ++i)
            {
                if(r_remaining[i] != nterms)
                {
                    r_remaining_spf[count] = r_remaining[i];
                    ++count;
                }
            }
        }

        for(const auto& mf_op : mf())
        {
            rtemp.clear();  mf_op.r().get(rtemp);
            for(size_type ri : rtemp)
            {
                nremoved += r_remaining[ri] != nterms ? 1 : 0;
                r_remaining[ri] = nterms; 
            }   
        }   
        {
            rtemp.resize(nterms - nremoved);
            size_type count = 0;
            for(size_type i = 0; i < nterms; ++i)
            {
                if(r_remaining[i] != nterms)
                {
                    rtemp[count] = r_remaining[i];
                    ++count;
                }
            }
            using std::swap;
            swap(rtemp, r_remaining);
        }
    }


protected:
    static void construct_spf_node_indices(const op_info_node& spf, const op_info_node& mf, const sop_operator<T, B>& op, index_node& n, node_type& res)
    {
        std::vector<size_type> r_remaining(op.nterms());   size_type count = 0; for(size_type& r : r_remaining){r = count; ++count;}
        std::vector<size_type> r_remaining_spf;
        remove_common_terms(spf, mf, op.nterms(), r_remaining_spf, r_remaining);

        size_type n_common_spf = spf().size();
        size_type n_common_mf = mf().size();
        size_type n_standard_size = r_remaining.size();

        size_type nterms = n_common_spf+n_common_mf + n_standard_size;
        CALL_AND_HANDLE(res().resize(n_common_spf, n_common_mf, n_standard_size), "Failed to resize operator node.");

        n().resize(nterms);

        CALL_AND_HANDLE(initialise_common_spf_upwards(spf, op, n, res), "Failed to initialise spf_common terms.");
        CALL_AND_HANDLE(initialise_common_mf_upwards(mf, op, r_remaining_spf, n, res), "Failed to initialise mff_common terms.");
        CALL_AND_HANDLE(initialise_standard_upwards(op, r_remaining, n, res), "Failed to initialise standard terms.");
    }


protected:
    static void initialise_common_spf_downwards(const sop_operator<T, B>& op, index_node& n, node_type& res)
    {
        size_type nterms = op.nterms();
        size_type cid = res.child_id();
        std::vector<size_type> rtemp;   rtemp.reserve(nterms);
        for(size_type i=0; i < res().ncommon_spf(); ++i)
        {
            //get the rvalues for this term in the node
            n()[i].get(rtemp);

            std::list<std::vector<size_type>> rres; 
            std::list<std::vector<size_type>> nids;

            using iterator_type = typename std::vector<minimal_index_array<size_type> >::const_iterator;

            size_type nch = n.parent().size();
            std::vector<iterator_type> ciiterators;         ciiterators.reserve(nch);
            std::vector<iterator_type> ciiterators_begin;   ciiterators_begin.reserve(nch);
            std::vector<iterator_type> ciiterators_end;     ciiterators_end.reserve(nch);

            ciiterators.push_back(n.parent()().cbegin());
            ciiterators_begin.push_back(n.parent()().cbegin());
            ciiterators_end.push_back(n.parent()().cend());

            for(size_type nu=0; nu < nch; ++nu)
            {
                if(nu != cid)
                {
                    ciiterators.push_back(n.parent()[nu]().cbegin());
                    ciiterators_begin.push_back(n.parent()[nu]().cbegin());
                    ciiterators_end.push_back(n.parent()[nu]().cend());
                }
            }       
            process_operator_iterators(ciiterators, ciiterators_begin, ciiterators_end, rtemp, rres, nids);

            //evaluate the different coefficients we need when accumulating the mean field operators.  Not here 
            //we don't flag whether an operator is part of a common mean field term.  And as such if there are 
            //terms that are completely common in the Hamiltonian we do not treat these as efficiently as possible.
            //However, such terms could be accumulated together at the start and so we will just deal with this 
            //not being as efficient as possible.
            size_type index = 0;

            res()[i].m_mf_index.resize(rres.size());
            res()[i].m_accum_coeff.resize(rres.size());
            res()[i].m_coeff = T(1);
            //now that we have the child indices that have the correct r values we go about setting everything up.
            for(auto z : zip(rres, nids))
            {   
                const auto& rinds = std::get<0>(z);  const auto& ni = std::get<1>(z);
                
                //set the coefficient array.  If there is only a single r term for this one, then it is constructed from a standard term and we  
                //should store the coefficient required.  If there is moer than one term, then it arises from an acculumated parent mf and common
                //sibling spfs and as a result we have already accumulated the coefficients for this terms, so we set its coefficient to 1.
                if(rinds.size() == 1){res()[i].m_accum_coeff[index] = op.coeff(rinds[0]);}
                else{res()[i].m_accum_coeff[index] = T(1);}


                size_type parent_index = ni[0];
                std::list<std::array<size_type, 2> > indexing;

                size_type mcount = 1;
                for(size_type nu=0; nu < nch; ++nu)
                {
                    if(nu != cid)
                    {
                        if(!res.parent()[nu]()[ni[mcount]].is_identity_spf())
                        {
                            indexing.push_back({{nu, ni[mcount]}});
                        }
                        ++mcount;
                    }
                }       
                res().common_spf_term(i).m_mf_index[index] = mf_index<size_type>{parent_index, {indexing.begin(), indexing.end()}};
                ++index;
            }
        }
    }
    
    static void initialise_common_mf_downwards(size_type nterms, index_node& n, node_type& res)
    {
        size_type cid = res.child_id();
        std::vector<size_type> rtemp;   rtemp.reserve(nterms);
        size_type skip = res().ncommon_spf();
        for(size_type i=0; i < res().ncommon_mf(); ++i)
        {
            //get the rvalues for this term in the node
            n()[i+skip].get(rtemp);
            
            std::list<std::vector<size_type>> rres; 
            std::list<std::vector<size_type>> nids;

            using iterator_type = typename std::vector<minimal_index_array<size_type> >::const_iterator;

            size_type nch = n.parent().size();
            std::vector<iterator_type> ciiterators;         ciiterators.reserve(nch);
            std::vector<iterator_type> ciiterators_begin;   ciiterators_begin.reserve(nch);
            std::vector<iterator_type> ciiterators_end;     ciiterators_end.reserve(nch);

            ciiterators.push_back(n.parent()().cbegin());
            ciiterators_begin.push_back(n.parent()().cbegin());
            ciiterators_end.push_back(n.parent()().cend());

            for(size_type nu=0; nu < nch; ++nu)
            {
                if(nu != cid)
                {
                    ciiterators.push_back(n.parent()[nu]().cbegin());
                    ciiterators_begin.push_back(n.parent()[nu]().cbegin());
                    ciiterators_end.push_back(n.parent()[nu]().cend());
                }
            }       

            //now due to the fact that the common mean fields cannot have more than one term.  All we need to do is search for any index in
            //the rtemp array not all.  As such we will only use its first element.
            std::vector<size_type> rt1(1);  rt1[0] = rtemp.front();
            process_operator_iterators(ciiterators, ciiterators_begin, ciiterators_end, rt1, rres, nids, 1);

            const auto& ni = nids.front();
            size_type parent_index = ni[0];
            std::list<std::array<size_type, 2> > indexing;

            size_type mcount = 1;
            for(size_type nu=0; nu < nch; ++nu)
            {
                if(nu != cid)
                {
                    if(!res.parent()[nu]()[ni[mcount]].is_identity_spf())
                    {
                        indexing.push_back({{nu, ni[mcount]}});
                    }
                    ++mcount;
                }
            }
    
            res().common_mf_term(i).m_mf_index.resize(1);
            res().common_mf_term(i).m_mf_index[0] = mf_index<size_type>{parent_index, {indexing.begin(), indexing.end()}};
        }
    }

    static void initialise_standard_downwards(index_node& n, node_type& res)
    {
        size_type cid = res.child_id();
        std::vector<size_type> rtemp;   rtemp.reserve(1);
        size_type skip = res().ncommon_spf() + res().ncommon_mf();
        for(size_type i=0; i < res().nstandard(); ++i)
        {
            //get the rvalues for this term in the node
            n()[i+skip].get(rtemp);
            
            std::list<std::vector<size_type>> rres; 
            std::list<std::vector<size_type>> nids;

            using iterator_type = typename std::vector<minimal_index_array<size_type> >::const_iterator;

            size_type nch = n.parent().size();
            std::vector<iterator_type> ciiterators;         ciiterators.reserve(nch);
            std::vector<iterator_type> ciiterators_begin;   ciiterators_begin.reserve(nch);
            std::vector<iterator_type> ciiterators_end;     ciiterators_end.reserve(nch);

            ciiterators.push_back(n.parent()().cbegin());
            ciiterators_begin.push_back(n.parent()().cbegin());
            ciiterators_end.push_back(n.parent()().cend());

            for(size_type nu=0; nu < nch; ++nu)
            {
                if(nu != cid)
                {
                    ciiterators.push_back(n.parent()[nu]().cbegin());
                    ciiterators_begin.push_back(n.parent()[nu]().cbegin());
                    ciiterators_end.push_back(n.parent()[nu]().cend());
                }
            }       
            process_operator_iterators(ciiterators, ciiterators_begin, ciiterators_end, rtemp, rres, nids, 1);

            const auto& ni = nids.front();
            size_type parent_index = ni[0];
            std::list<std::array<size_type, 2> > indexing;

            bool mf_is_identity = res.parent()()[ni[0]].is_identity_mf();
            size_type mcount = 1;
            for(size_type nu=0; nu < nch; ++nu)
            {
                if(nu != cid)
                {
                    if(!res.parent()[nu]()[ni[mcount]].is_identity_spf())
                    {
                        mf_is_identity = false;
                        indexing.push_back({{nu, ni[mcount]}});
                    }
                    ++mcount;
                }
            }
    
            res().standard_term(i).m_mf_index.resize(1);
            res().standard_term(i).m_mf_index[0] = mf_index<size_type>{parent_index, {indexing.begin(), indexing.end()}};
            res().standard_term(i).m_is_identity_mf = mf_is_identity;
        }
    }
protected:
    static void construct_mf_node_indices(const sop_operator<T, B>& op, index_node& n, node_type& res)
    {
        size_type nterms = op.nterms();
        //If we are at the root node the mean field operator is the identity so we simply flag each term as the identity.
        //Additionally, there are only standard terms if we have a single term in the operator.  There are only 
        //spf_common terms if the operator contains repeated terms
        if(n.is_root())
        {
            for(size_type i=0; i < res().nterms(); ++i){res()[i].m_is_identity_mf = true;}
        }
        else
        {
            CALL_AND_HANDLE(initialise_common_spf_downwards(op, n, res), "Failed to initialise spf_common terms.");
            CALL_AND_HANDLE(initialise_common_mf_downwards(nterms, n, res), "Failed to initialise mff_common terms.");
            CALL_AND_HANDLE(initialise_standard_downwards(n, res), "Failed to initialise standard terms.");
        }
    }

protected:
    //ignore that this is bad pracitce and I have just copied and pasted these functions and made some minimal modifications
    template <typename IterArr>
    static void advance_iterators(IterArr& ciiterators, const IterArr& ciiterators_begin, const IterArr& ciiterators_end, size_type uind, std::vector<bool>& requires_update, std::vector<size_type>& hindex, std::vector<std::vector<size_type>>& union_temporary)
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
    static void process_operator_iterators(IterArr& ciiterators, const IterArr& ciiterators_begin, const IterArr& ciiterators_end, const std::vector<size_type>& riv, std::list<std::vector<size_type>>& rres, std::list<std::vector<size_type>> & nids, size_type nterms_to_find = 0)
    {
        size_type maxterms = riv.size();
        size_type nch = ciiterators.size();
        std::vector<size_type> hindex(nch); std::fill(hindex.begin(), hindex.end(), 0);

        std::vector<std::vector<size_type>> union_temporary(nch);
        for(size_type i=0; i < nch; ++i){union_temporary.reserve(maxterms);}

        std::vector<std::vector<size_type>> temp_vals(nch);
        for(size_type i=0; i < nch; ++i){temp_vals.reserve(maxterms);}

        std::vector<size_type> nid_term(nch);

        std::vector<bool> requires_update(nch); std::fill(requires_update.begin(), requires_update.end(), true);        

        while(ciiterators[0] != ciiterators_end[0])
        {
            if(requires_update[0])
            {
                ciiterators[0]->get(temp_vals[0]);
                union_temporary[0].clear();
                std::set_intersection(temp_vals[0].begin(), temp_vals[0].end(), riv.begin(), riv.end(), std::back_inserter(union_temporary[0]));
                requires_update[0] = false;
            }
            
            bool keep_running = true;
            for(size_type i=1; (i < nch) && keep_running; ++i)
            {
                if(requires_update[i])
                {
                    union_temporary[i].clear();
                    ciiterators[i]->get(temp_vals[i]);
                    if(union_temporary[i-1].size() > 1 && temp_vals[i].size() > 1)
                    {
                        std::set_intersection(temp_vals[i].begin(), temp_vals[i].end(), union_temporary[i-1].begin(), union_temporary[i-1].end(), std::back_inserter(union_temporary[i]));
                    }
                    else if(union_temporary[i-1].size() == 1 && temp_vals[i].size() == 1 )
                    {
                        if(union_temporary[i-1].back() == temp_vals[i].back()){union_temporary[i].push_back(union_temporary[i-1].back());}
                    }
                    else if(union_temporary[i-1].size() == 1)
                    {
                        if(std::binary_search(temp_vals[i].begin(), temp_vals[i].end(), union_temporary[i-1].back())){union_temporary[i].push_back(union_temporary[i-1].back());}
                    }
                    else if(temp_vals[i].size() == 1)
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
                if(union_temporary.back().size() > 0)
                {
                    rres.push_back(union_temporary.back());
    
                    for(size_type i=0; i < nch; ++i)
                    {
                        nid_term[i] = hindex[i];
                    }
                    nids.push_back(nid_term);
                    if(nterms_to_find > 0 && rres.size() == nterms_to_find){return;}
                }
                advance_iterators(ciiterators, ciiterators_begin, ciiterators_end, uind, requires_update, hindex, union_temporary);
            }
        }
    }
protected:
    tree_type m_op;
};  //class operator_container<T, B> 

}   //namespace ttns

#endif  //HTTENSOR_OPERATOR_CONTAINER_HPP//

