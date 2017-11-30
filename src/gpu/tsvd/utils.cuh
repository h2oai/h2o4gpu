#pragma once
#include "cublas_v2.h"
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <sstream>
#include <thrust/system/cuda/error.h>
#include <cusolver_common.h>
#include <ctime>
#include <cusparse.h>

// TODO change this to h2o4gpu and move to gpu folder
namespace tsvd
{
#define tsvd_error(x) error(x, __FILE__, __LINE__);

	inline void error(const char* e, const char* file, int line)
	{
		std::stringstream ss;
		ss << e << " - " << file << "(" << line << ")";
		//throw error_text;
		std::cerr << ss.str() << std::endl;
		exit(-1);
	}

#define tsvd_check(condition, msg) check(condition, msg, __FILE__, __LINE__);

	inline void check(bool val, const char* e, const char* file, int line)
	{
		if (!val)
		{
			error(e, file, line);
		}
	}


#define safe_cuda(ans) throw_on_cuda_error((ans), __FILE__, __LINE__)

	inline cudaError_t throw_on_cuda_error(cudaError_t code, const char* file, int line)
	{
		if (code != cudaSuccess)
		{
			std::stringstream ss;
			ss << cudaGetErrorString(code) << " - " << file << "(" << line << ")";
			//throw error_text;
			std::cerr << ss.str() << std::endl;
			exit(-1);
		}

		return code;
	}

	inline static const char* cublasGetErrorEnum(cublasStatus_t error)
	{
		switch (error)
		{
		case CUBLAS_STATUS_SUCCESS:
			return "CUBLAS_STATUS_SUCCESS";

		case CUBLAS_STATUS_NOT_INITIALIZED:
			return "CUBLAS_STATUS_NOT_INITIALIZED";

		case CUBLAS_STATUS_ALLOC_FAILED:
			return "CUBLAS_STATUS_ALLOC_FAILED";

		case CUBLAS_STATUS_INVALID_VALUE:
			return "CUBLAS_STATUS_INVALID_VALUE";

		case CUBLAS_STATUS_ARCH_MISMATCH:
			return "CUBLAS_STATUS_ARCH_MISMATCH";

		case CUBLAS_STATUS_MAPPING_ERROR:
			return "CUBLAS_STATUS_MAPPING_ERROR";

		case CUBLAS_STATUS_EXECUTION_FAILED:
			return "CUBLAS_STATUS_EXECUTION_FAILED";

		case CUBLAS_STATUS_INTERNAL_ERROR:
			return "CUBLAS_STATUS_INTERNAL_ERROR";
		}

		return "<unknown>";
	}

#define safe_cublas(ans) throw_on_cublas_error((ans), __FILE__, __LINE__)

	inline cublasStatus_t throw_on_cublas_error(cublasStatus_t status, const char* file, int line)
	{
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			std::stringstream ss;
			ss << cublasGetErrorEnum(status) << " - " << file << "(" << line << ")";
			std::string error_text;
			ss >> error_text;
			//throw error_text;
			std::cerr << error_text << std::endl;
			exit(-1);
		}

		return status;
	}

#define safe_cusolver(ans) throw_on_cusolver_error((ans), __FILE__, __LINE__)

	inline cusolverStatus_t throw_on_cusolver_error(cusolverStatus_t status, const char* file, int line)
	{
		if (status != CUSOLVER_STATUS_SUCCESS)
		{
			std::stringstream ss;
			ss << "cusolver error: " << file << "(" << line << ")";
			std::string error_text;
			ss >> error_text;
			//throw error_text;
			std::cerr << error_text << std::endl;
			exit(-1);
		}

		return status;
	}

#define safe_cusparse(ans) throw_on_cusparse_error((ans), __FILE__, __LINE__)

	inline cusparseStatus_t throw_on_cusparse_error(cusparseStatus_t status, const char* file, int line)
	{
		if (status != CUSPARSE_STATUS_SUCCESS)
		{
			std::stringstream ss;
			ss << "cusparse error: " << file << "(" << line << ")";
			std::string error_text;
			ss >> error_text;
			//throw error_text;
			std::cerr << error_text << std::endl;
			exit(-1);
		}

		return status;
	}
	template <typename T>
	void print(thrust::device_vector<T>& v)
	{
		thrust::device_vector<T> h_v = v;
		for (int i = 0; i < h_v.size(); i++)
		{
			std::cout << h_v[i] << " ";
		}
		std::cout << "\n";
	}

	#define TIMERS
	struct Timer
	{
		volatile double start;
		Timer() { reset(); }

		double seconds_now()
		{
#ifdef _WIN32
			static LARGE_INTEGER s_frequency;
			QueryPerformanceFrequency(&s_frequency);
			LARGE_INTEGER now;
			QueryPerformanceCounter(&now);
			return static_cast<double>(now.QuadPart) / s_frequency.QuadPart;
#else
			return 0;
#endif
		}

		void reset()
		{
#ifdef _WIN32
			_ReadWriteBarrier();
			start = seconds_now();
#endif
		}

		double elapsed()
		{
#ifdef _WIN32
			_ReadWriteBarrier();
			return seconds_now() - start;
#else
			return 0;
#endif
		}

		void printElapsed(std::string label)
		{
#ifdef TIMERS
			safe_cuda(cudaDeviceSynchronize());
			printf("%s:\t %1.4fs\n", label.c_str(), elapsed());
#endif
		}
	};

	inline double clocks_to_s(clock_t t)
	{
		return (double)t / CLOCKS_PER_SEC;
	}

	struct sqr_op
	{
		__device__ float operator()(float val) const
		{
			return val * val;
		};
	};

	struct sqrt_op
	{
		__device__ float operator()(float val) const
		{
			return sqrt(val);
		};
	};

	/*
	* Range iterator
	*/

	class range
	{
	public:
		class iterator
		{
			friend class range;

		public:
			__host__ __device__ int64_t operator*() const { return i_; }
			__host__ __device__ const iterator& operator++()
			{
				i_ += step_;
				return *this;
			}

			__host__ __device__ iterator operator++(int)
			{
				iterator copy(*this);
				i_ += step_;
				return copy;
			}

			__host__ __device__ bool operator==(const iterator& other) const
			{
				return i_ >= other.i_;
			}

			__host__ __device__ bool operator!=(const iterator& other) const
			{
				return i_ < other.i_;
			}

			__host__ __device__ void step(int s) { step_ = s; }

		protected:
			__host__ __device__ explicit iterator(int64_t start) : i_(start)
			{
			}

		public:
			uint64_t i_;
			int step_ = 1;
		};

		__host__ __device__ iterator begin() const { return begin_; }
		__host__ __device__ iterator end() const { return end_; }
		__host__ __device__ range(int64_t begin, int64_t end)
			: begin_(begin), end_(end)
		{
		}

		__host__ __device__ void step(int s) { begin_.step(s); }

	private:
		iterator begin_;
		iterator end_;
	};

	template <typename T>
	__device__ range grid_stride_range(T begin, T end)
	{
		begin += blockDim.x * blockIdx.x + threadIdx.x;
		range r(begin, end);
		r.step(gridDim.x * blockDim.x);
		return r;
	}

	template <typename T>
	__device__ range block_stride_range(T begin, T end)
	{
		begin += threadIdx.x;
		range r(begin, end);
		r.step(blockDim.x);
		return r;
	}


	// Threadblock iterates over range, filling with value
	template <typename IterT, typename ValueT>
	__device__ void block_fill(IterT begin, size_t n, ValueT value)
	{
		for (auto i : block_stride_range(static_cast<size_t>(0), n))
		{
			begin[i] = value;
		}
	}

	template <typename SrcIterT, typename DestIterT>
	__device__ void block_copy(SrcIterT src, DestIterT dest, size_t n)
	{
		for (auto i : block_stride_range(static_cast<size_t>(0), n))
		{
			dest[i] = src[i];
		}
	}

	template <typename T>
	void tprint(thrust::device_vector<T> &v, const char * label = "")
	{
		if (strlen(label))
		{
			printf("%s: ", label);
		}

		thrust::host_vector<T> h_v = v;
		for (int i = 0; i < v.size(); i++)
		{
			std::cout << h_v[i] << " ";
		}
		std::cout << "\n";
	}

	// Keep track of cub library device allocation
	struct CubMemory {
		void *d_temp_storage;
		size_t temp_storage_bytes;

		CubMemory() : d_temp_storage(NULL), temp_storage_bytes(0) {}

		~CubMemory() {
				this->Free();
		}

		void Free()
		{
			if (d_temp_storage != NULL) {
				safe_cuda(cudaFree(d_temp_storage));
			}
			temp_storage_bytes = 0;
		}

		void LazyAllocate( size_t bytes) {
			if (bytes > temp_storage_bytes){
				this->Free();
				temp_storage_bytes = bytes;
				safe_cuda(cudaMalloc(&d_temp_storage, temp_storage_bytes));
			}
		}

		bool IsAllocated() { return d_temp_storage != NULL; }
	};

	inline void generate_column_segments(thrust::device_vector<int>& column_segments, int col_size)
	{
		auto counting = thrust::make_counting_iterator(0);
		thrust::transform(counting, counting + column_segments.size(), column_segments.begin(), [=]__device__(int idx)
		                  {
			                  return idx * col_size;
		                  });
	}
}
