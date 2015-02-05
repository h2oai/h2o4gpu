#ifndef PROJECTOR_PROJECTOR_H_ 
#define PROJECTOR_PROJECTOR_H_ 

#include <memory>

namespace pogs {

// Minimizes ||Ax - y0||^2  + s ||x - x0||^2
template <typename T, typename M>
class Projector {
 protected:
  bool _done_init;

  // TODO: Implement copy constructor.
  void *_info;

 public:
  Projector() : _done_init(false), _info(0) { };
  virtual ~Projector() { };
  
  virtual int Init() = 0;

  virtual int Free() = 0;

  // TODO tolerance for solve. 
  virtual int Project(const T *x0, const T *y0, T s, T *x, T *y) = 0;
  
  bool IsInit() { return _done_init; }
};

}  // namespace pogs

#endif  // PROJECTOR_PROJECTOR_H_ 

