#ifndef HTUCKER_HT_TENSOR_NODE_TRAITS_HPP
#define HTUCKER_HT_TENSOR_NODE_TRAITS_HPP

#include <linalg/linalg.hpp>

#include <type_traits>

namespace ttns
{

namespace node_data_traits
{
    //assignment traits for the httensor node data object
    template <typename T, typename U, typename backend1, typename backend2>
    struct assignment_traits<httensor_node_data<T, backend1>, httensor_node_data<U, backend2>>
    {
        using hdata = httensor_node_data<T, backend1>;
        using hdata2 = httensor_node_data<U, backend2>;
        using is_applicable = std::is_convertible<U, T>;

        inline typename std::enable_if<std::is_convertible<U, T>::value, void>::type operator()(hdata& o,  const hdata2& i)
        {
            CALL_AND_RETHROW(o = i);
        }
    };

    //assignment traits for the httensor node data object
    template <typename T, typename U, typename backend1, typename backend2>
    struct assignment_traits<httensor_node_data<T, backend1>, linalg::matrix<U, backend2>>
    {
        using hdata = httensor_node_data<T, backend1>;
        using mat = linalg::matrix<U, backend2>;

        using is_applicable = std::is_convertible<U, T>;

        inline typename std::enable_if<std::is_convertible<U, T>::value, void>::type operator()(hdata& o,  const mat& i)
        {
            CALL_AND_RETHROW(o = i);
        }
    };


    //resize traits for the httensor node data object
    template <typename T, typename U, typename backend1, typename backend2>
    struct resize_traits<httensor_node_data<T, backend1>, httensor_node_data<U, backend2> >
    {
        template <typename V, typename backend> using hdata = httensor_node_data<V, backend>;

        using is_applicable = std::true_type;

        inline void operator()(hdata<T, backend1>& o,  const hdata<U, backend2>& i)
        {
            CALL_AND_RETHROW(o.resize(i.hrank(), i.dims()));
        }
    };

    //resize traits for the httensor node data object
    template <typename T, typename U, typename backend1, typename backend2>
    struct reallocate_traits<httensor_node_data<T, backend1>, httensor_node_data<U, backend2> >
    {
        template <typename V, typename backend> using hdata = httensor_node_data<V, backend>;

        using is_applicable = std::true_type;

        inline void operator()(hdata<T, backend1>& o,  const hdata<U, backend2>& i)
        {
            CALL_AND_RETHROW(o.reallocate(i.max_hrank(), i.max_dims()));
        }
    };

    //size comparison traits for the httensor node data object
    template <typename T, typename U, typename backend1, typename backend2>
    struct size_comparison_traits<httensor_node_data<T, backend1>, httensor_node_data<U, backend2> >
    {
        template <typename V, typename backend> using hdata = httensor_node_data<V, backend>;

        using is_applicable = std::true_type;

        inline bool operator()(const hdata<T, backend1>& o,  const hdata<U, backend2>& i)
        {
            if(o.nmodes() != i.nmodes()){return false;}
            else
            {
                return ((o.shape() == i.shape()) && (o.dims() == i.dims()));
            }
        }
    };

    template <typename T, typename U, typename backend1, typename backend2>
    struct size_comparison_traits<httensor_node_data<T, backend1>, linalg::matrix<U, backend2> >
    {
        using hdata = httensor_node_data<T, backend1>;
        using mat = linalg::matrix<U, backend2>;

        using is_applicable = std::true_type;

        inline bool operator()(const hdata& o,  const mat& i)
        {
            return (o.shape() == i.shape());
        }
    };

    //clear traits for the httensor node data object
    template <typename T, typename backend>
    struct clear_traits<httensor_node_data<T, backend> > 
    {
        void operator()(httensor_node_data<T, backend>& t){CALL_AND_RETHROW(t.clear());}
    };
}

}

#endif  //HTUCKER_TENSOR_NODE_TRAITS_HPP//

