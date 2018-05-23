/*!
 * Modifications Copyright 2017-2018 H2O.ai, Inc.
 */
#ifndef PROJECTOR_PROJECTOR_H_
#define PROJECTOR_PROJECTOR_H_ 

namespace h2o4gpu {

// Minimizes ||Ax - y0||^2  + s ||x - x0||^2
template <typename T, typename M>
class Projector {
 protected:
  bool _done_init;

  void *_info;

 public:
  Projector() : _done_init(false), _info(0) { };
  virtual ~Projector() { };
  
  virtual int Init() = 0;

  virtual int Project(const T *x0, const T *y0, T s, T *x, T *y, T tol) = 0;
  
  bool IsInit() { return _done_init; }
};

}  // namespace h2o4gpu

#endif  // PROJECTOR_PROJECTOR_H_ 

