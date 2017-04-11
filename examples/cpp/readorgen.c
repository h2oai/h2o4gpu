


std::default_random_engine generator;
std::uniform_real_distribution<T> u_dist(static_cast<T>(0),
                                         static_cast<T>(1));
std::normal_distribution<T> n_dist(static_cast<T>(0),
                                   static_cast<T>(1));


// READ-IN DATA
if(generate==0){
  size_t R=m; // rows
  size_t C=n+1; // columns
  FILE * file = fopen("train.txt","rt");
  if(file==NULL){
    fprintf(stderr,"Cannot read file.\n");
    exit(1);
  }
  else{
    // Expect space-separated file with response in last column, no header
    double v;
    int count=0;
    for (unsigned int i = 0; i < R; ++i){// rows
      for (unsigned int j = 0; j < C-1; ++j){ // columns
        count += fscanf(file,"%lf",&v);
        A[i*n+j]=(float)v;
      }
      count += fscanf(file,"%lf",&v);
      b[i]=(float)v;
    }
    for (unsigned int i = 0; i < m * n; ++i){
      if(!std::isfinite(A[i])) fprintf(stderr,"NF: A[%d]=%g\n",i,A[i]);
    }
    for (unsigned int i = 0; i < m; ++i){
      if(!std::isfinite(b[i])) fprintf(stderr,"b[%d]=%g\n",i,b[i]);
    }
  }
 }
 else{
   // GENERATE DATA
//#pragma omp parallel for //FIXME - test this
   for (unsigned int i = 0; i < m * n; ++i)
     A[i] = n_dist(generator);

   std::vector<T> x_true(n);
   for (unsigned int i = 0; i < n; ++i)
     x_true[i] = u_dist(generator) < static_cast<T>(0.8)
       ? static_cast<T>(0) : n_dist(generator) / static_cast<T>(std::sqrt(n));

#ifdef _OPENMP
#pragma omp parallel for
#endif
   for (unsigned int i = 0; i < m; ++i) // rows
     for (unsigned int j = 0; j < n; ++j) // columns
       b[i] += A[i * n + j] * x_true[j]; // C(0-indexed) row-major order
   // b[i] += A[i + j * m] * x_true[j];

   for (unsigned int i = 0; i < m; ++i)
     b[i] += static_cast<T>(0.5) * n_dist(generator);

 }
