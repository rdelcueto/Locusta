#ifndef _CURAND_HELPER_H_
#define _CURAND_HELPER_H_

#include <stdio.h>

#define CurandSafeCall(err)                     \
  {                                             \
    if(CURAND_STATUS_SUCCESS != err)            \
      {                                         \
        fprintf(stderr,                         \
                "curand error at %i:%s\n",      \
                __LINE__, __FILE__);            \
        exit(EXIT_FAILURE);                     \
      }                                         \
  }

#endif /* _CURAND_HELPER_H_ */
