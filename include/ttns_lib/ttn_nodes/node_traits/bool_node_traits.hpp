#ifndef HTUCKER_BOOL_NODE_TRAITS_HPP
#define HTUCKER_BOOL_NODE_TRAITS_HPP

namespace ttns
{

namespace node_data_traits
{
    //assignment traits for the tensor and matrix objects
    template <>
    struct assignment_traits<bool, bool>
    {
        using is_applicable = std::true_type;

        inline void operator()(bool& o,  const bool& i){o=i;}
    };

    //resize traits for tensor and matrix objects
    template <typename ... Args>
    struct resize_traits<bool, Args...>
    {
        using is_applicable = std::true_type;
        inline void operator()(bool& /* o */, const Args&... /* args */){}
    };

    template <typename ... Args>
    struct reallocate_traits<bool, Args...>
    {
        using is_applicable = std::true_type;
        inline void operator()(bool& /* o */, const Args&... /* args */){}
    };

    //size comparison traits for the tensor and matrix objects
    template <typename ... Args>
    struct size_comparison_traits<bool, Args...>
    {
        using is_applicable = std::true_type;

        inline bool operator()(const bool& /* o */, const Args&... /* i */){return true;}
    };
}

}

#endif  //HTUCKER_BOOL_NODE_TRAITS_HPP//

