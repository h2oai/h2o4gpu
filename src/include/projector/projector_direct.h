#ifndef PROJECTOR_PROJECTOR_DIRECT_H_ 
#define PROJECTOR_PROJECTOR_DIRECT_H_ 

#include "projector/projector.h"

namespace h2o4gpu {

// Minimizes ||Ax - y0||^2  + s ||x - x0||^2
template <typename T, typename M>
class ProjectorDirect : Projector<T, M> {
 public:
  int _wDev;
 private:
  const M& _A;


  // Get rid of copy constructor and assignment operator.
  ProjectorDirect(const Projector<T, M>& A);
  ProjectorDirect<M, T>& operator=(const ProjectorDirect<T, M>& P);

 public:
  ProjectorDirect(int wDev, const M& A);
  ProjectorDirect(const M& A);
  ~ProjectorDirect();
  
  int Init();

  int Project(const T *x0, const T *y0, T s, T *x, T *y, T tol);
};

}  // namespace h2o4gpu

#endif  // PROJECTOR_PROJECTOR_DIRECT_H_ 

