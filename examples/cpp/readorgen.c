

// choose to generate or read-in data
  int generate=1;

    std::default_random_engine generator;
    std::uniform_real_distribution<T> u_dist(static_cast<T>(0),
					     static_cast<T>(1));
    std::normal_distribution<T> n_dist(static_cast<T>(0),
				       static_cast<T>(1));


// READ-IN DATA
  double t0 = timer<double>();
  if(generate==0){
    int TARGETCOL=42;
    int dAm=m; // rows
    int dAn=n+1; // columns
    FILE * file = fopen("/tmp/train.txt","rt");
    if(file==NULL){
      fprintf(stderr,"Cannot read file.\n");
      exit(1);
    }
    else{

      // tail -n +2 /tmp/train.csv > /tmp/trainnonhead.csv
      // sed 's/,/ /g' /tmp/trainnonhead.csv > /tmp/train.txt
      
      fprintf(stdout,"BEGIN READ DATA\n");
      double dum;
      int Ai,Aj;
      for (unsigned int i = 0; i < dAm; ++i){// rows
	Ai=i;
	//fprintf(stderr,"row=%d\n",i);
	Aj=0;
	for (unsigned int j = 0; j < dAn; ++j){ // columns
	  if(j!=TARGETCOL-1){
	    Aj++;
	    fscanf(file,"%lf",&dum);
	    A[Ai*n+Aj]=dum;
	  }
	  else if(j==TARGETCOL-1){
	    fscanf(file,"%lf",&dum);
	    b[Ai]=-dum;
	  }
	}
      }
      for (unsigned int i = 0; i < m * n; ++i){
	//fprintf(stderr,"A[%d]=%g\n",i,A[i]);
	if(!std::isfinite(A[i])) fprintf(stderr,"NF: A[%d]=%g\n",i,A[i]);
      }
      for (unsigned int i = 0; i < m; ++i){
	if(!std::isfinite(b[i])) fprintf(stderr,"b[%d]=%g\n",i,b[i]);
      }
    }
  }
  else{
    // GENERATE DATA
    fprintf(stdout,"BEGIN FILL DATA\n");
    for (unsigned int i = 0; i < m * n; ++i)
      A[i] = n_dist(generator);

    std::vector<T> x_true(n);
    for (unsigned int i = 0; i < n; ++i)
      x_true[i] = u_dist(generator) < static_cast<T>(0.8)
				      ? static_cast<T>(0) : n_dist(generator) / static_cast<T>(std::sqrt(n));

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (unsigned int i = 0; i < m; ++i)
      for (unsigned int j = 0; j < n; ++j)
	b[i] += A[i * n + j] * x_true[j];
    // b[i] += A[i + j * m] * x_true[j];

    for (unsigned int i = 0; i < m; ++i)
      b[i] += static_cast<T>(0.5) * n_dist(generator);

  }
