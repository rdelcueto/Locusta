#ifndef _BOUND_MAPPING_CPU_H_
#define _BOUND_MAPPING_CPU_H_

#include <limits>

namespace locusta {

    template<typename TFloat>
        inline void bound_mapping(const uint32_t Bound_Mapping_Method,
                                  const TFloat &u,
                                  const TFloat &l,
                                  TFloat &x)
    {
        switch (Bound_Mapping_Method)
        {
        case 0: // Out of bounds Crop
            x = x > u ? u : x < l ? l : x;
            break;
        case 1: // Out of bounds Mirror
            x = x > u ? (2*u - x) : x < l ? (2*l - x) : x;
            break;
        case 2: // Out of bounds Error
            if ( x > u || x < l )
            {
                x = std::numeric_limits<TFloat>::quiet_NaN();
                std::cerr << "Out of bounds gene!" << std::endl;
            }
            break;
        }
        return;
    }
}

#endif /* _BOUND_MAPPING_CPU_H_ */
