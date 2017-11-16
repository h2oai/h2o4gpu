#!/usr/bin/groovy
// TOOD: rename to @Library('h2o-jenkins-pipeline-lib') _
@Library('test-shared-library') _

import ai.h2o.ci.Utils

def utilsLib = new Utils()

def SAFE_CHANGE_ID = changeId()
def CONTAINER_NAME

String changeId() {
    if (env.CHANGE_ID) {
        return "-${env.CHANGE_ID}".toString()
    }
    return "-master"
}



pipeline {
    agent none

    // Setup job options
    options {
        ansiColor('xterm')
        timestamps()
        timeout(time: 120, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '10'))
        disableConcurrentBuilds()
        skipDefaultCheckout()
    }

    environment {
        MAKE_OPTS = "-s CI=1" // -s: silent mode
    }

    stages {

        stage('Build on Linux nccl CUDA8') {
            agent {
                label "nvidia-docker && (mr-dl11||mr-dl16||mr-dl10)"
            }

            steps {
                dumpInfo 'Linux Build Info'
                // Do checkout
                retryWithTimeout(100 /* seconds */, 3 /* retries */) {
                    deleteDir()
                    checkout([
                            $class                           : 'GitSCM',
                            branches                         : scm.branches,
                            doGenerateSubmoduleConfigurations: false,
                            extensions                       : scm.extensions + [[$class: 'SubmoduleOption', disableSubmodules: true, recursiveSubmodules: false, reference: '', trackingSubmodules: false, shallow: true]],
                            submoduleCfg                     : [],
                            userRemoteConfigs                : scm.userRemoteConfigs])
                }

                script {
                    def extratag = "_nccl_cuda8"
                    CONTAINER_NAME = "h2o4gpu-${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    echo "CONTAINER_NAME = ${CONTAINER_NAME}"
                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        try {
                            sh """
                                    nvidia-docker build  -t opsh2oai/h2o4gpu-${extratag}-build -f Dockerfile-build --build-arg cuda=nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04 .
                                    nvidia-docker run --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-${extratag}-build
                                    nvidia-docker exec ${CONTAINER_NAME} rm -rf data
                                    nvidia-docker exec ${CONTAINER_NAME} ln -s /data ./data
                                    nvidia-docker exec ${CONTAINER_NAME} rm -rf open_data
                                    nvidia-docker exec ${CONTAINER_NAME} ln -s /open_data ./open_data
                                    nvidia-docker exec ${
                                CONTAINER_NAME
                            } bash -c 'eval \"\$(/root/.pyenv/bin/pyenv init -)\" ; /root/.pyenv/bin/pyenv global 3.6.1; ./scripts/gitshallow_submodules.sh; make ${
                                env.MAKE_OPTS
                            } AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} fullinstalljenkins${extratag} ; rm -rf build/VERSION.txt ; make build/VERSION.txt'
                                """
                            stash includes: 'src/interface_py/dist/*.whl', name: 'linux_whl'
                            stash includes: 'build/VERSION.txt', name: 'version_info'
                            // Archive artifacts
                            arch 'src/interface_py/dist/*.whl'
                        } finally {
                            sh "nvidia-docker stop ${CONTAINER_NAME}"
                        }
                    }
                }
            }
        }



        stage('Full Test on Linux nccl CUDA8') {
            agent {
                label "gpu && nvidia-docker && (mr-dl11||mr-dl16||mr-dl10)"
            }
            steps {
                dumpInfo 'Linux Test Info'
                // Get source code (should put tests into wheel, then wouldn't have to checkout)
                retryWithTimeout(100 /* seconds */, 3 /* retries */) {
                    checkout scm
                }
                unstash 'linux_whl'
                def extratag = "_nccl_cuda8"
                script {
                    echo "CONTAINER_NAME = ${CONTAINER_NAME}"
                    try {
                        sh """
                            nvidia-docker run  --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-${extratag}-build
                            nvidia-docker exec ${CONTAINER_NAME} rm -rf data
                            nvidia-docker exec ${CONTAINER_NAME} ln -s /data ./data
                            nvidia-docker exec ${CONTAINER_NAME} rm -rf open_data
                            nvidia-docker exec ${CONTAINER_NAME} ln -s /open_data ./open_data
                            nvidia-docker exec ${CONTAINER_NAME} rm -rf py3nvml
                            nvidia-docker exec ${CONTAINER_NAME} bash -c 'export HOME=`pwd`; eval \"\$(/root/.pyenv/bin/pyenv init -)\"  ; /root/.pyenv/bin/pyenv global 3.6.1; pip install `find src/interface_py/dist -name "*h2o4gpu*.whl"`; make dotest'
                        """
                    } finally {
                        sh """
                            nvidia-docker stop ${CONTAINER_NAME}
                        """
                        arch 'tmp/*.log'
                        junit testResults: 'build/test-reports/*.xml', keepLongStdio: true, allowEmptyResults: false
                        deleteDir()
                    }
                }
            }
        }
        stage('Pylint on Linux nccl CUDA8') {
            agent {
                label "gpu && nvidia-docker && (mr-dl11||mr-dl16||mr-dl10)"
            }

            def extratag = "_nccl_cuda8"
            steps {
                dumpInfo 'Linux Pylint Info'
                checkout([
                        $class                           : 'GitSCM',
                        branches                         : scm.branches,
                        doGenerateSubmoduleConfigurations: false,
                        extensions                       : scm.extensions + [[$class: 'SubmoduleOption', disableSubmodules: true, recursiveSubmodules: false, reference: '', trackingSubmodules: false, shallow: true]],
                        submoduleCfg                     : [],
                        userRemoteConfigs                : scm.userRemoteConfigs])
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                    sh """
                            nvidia-docker build  -t opsh2oai/h2o4gpu-${extratag}-build -f Dockerfile-build  --build-arg cuda=nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04 .
                            nvidia-docker run  --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-${extratag}-build
                            nvidia-docker exec ${CONTAINER_NAME} touch src/interface_py/h2o4gpu/__init__.py
                            nvidia-docker exec ${CONTAINER_NAME} bash -c 'eval \"\$(/root/.pyenv/bin/pyenv init -)\"  ;  /root/.pyenv/bin/pyenv global 3.6.1; make pylint'
                            nvidia-docker stop ${CONTAINER_NAME}
                        """
                }
            }
        }


        stage('Publish to S3 nccl CUDA8') {
            agent {
                label "linux"
            }

            steps {
                unstash 'version_info'
                def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                def extratag = "_nccl_cuda8"
                unstash 'linux_whl'
                sh 'echo "Stashed files:" && ls -l src/interface_py/dist/'
                def artifactId = h2o4gpu
                def artifact = "${artifactId}-${versionTag}-py36-none-any.whl"
                def localArtifact = "src/interface_py/dist/${artifact}"

                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                    script {
                        s3up_simple(${versionTag}, ${extratag}, ${artifactId}, ${artifact}, ${localArtifact})
                    }
                }
                }
            }
        }

        stage('Build Runtime Docker for nccl CUDA8') {
            agent {
                label "nvidia-docker && (mr-dl11||mr-dl16||mr-dl10)"
            }

            steps {
                dumpInfo 'Linux Build Info'
                // Do checkout
                retryWithTimeout(100 /* seconds */, 3 /* retries */) {
                    deleteDir()
                    checkout([
                            $class                           : 'GitSCM',
                            branches                         : scm.branches,
                            doGenerateSubmoduleConfigurations: false,
                            extensions                       : scm.extensions + [[$class: 'SubmoduleOption', disableSubmodules: true, recursiveSubmodules: false, reference: '', trackingSubmodules: false, shallow: true]],
                            submoduleCfg                     : [],
                            userRemoteConfigs                : scm.userRemoteConfigs])
                }
                script {
                    sh """
                        mkdir -p build ; rm -rf build/VERSION.txt
                    """
                }
                unstash 'version_info'
                sh 'echo "Stashed version file:" && ls -l build/'
                script {
                    def extratag = "_nccl_cuda8"
                    def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                    //CONTAINER_NAME = "h2o4gpu-${versionTag}${extratag}-runtime${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    CONTAINER_NAME = "h2o4gpu-runtime"
                    echo "CONTAINER_NAME = ${CONTAINER_NAME}"
                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        sh """
                                nvidia-docker build -t opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime:latest -f Dockerfile-runtime --build-arg cuda=nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04 --build-arg wheel=${versionTag}${extratag}/h2o4gpu-${versionTag}-py36-none-any.whl .
                                nvidia-docker run  --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime
                                nvidia-docker exec ${CONTAINER_NAME} rm -rf data
                                nvidia-docker exec ${CONTAINER_NAME} ln -s /data ./data
                                nvidia-docker exec ${CONTAINER_NAME} rm -rf open_data
                                nvidia-docker exec ${CONTAINER_NAME} ln -s /open_data ./open_data
                                nvidia-docker exec ${CONTAINER_NAME} bash -c '. /h2o4gpu_env/bin/activate ; pip freeze'
                                nvidia-docker exec ${CONTAINER_NAME} bash -c '. /h2o4gpu_env/bin/activate ; cd /jupyter/demos ; python -c "exec(\\"from sklearn.datasets import fetch_covtype\\ncov = fetch_covtype()\\")"'
                                nvidia-docker exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; cp /open_data/creditcard.csv .'
                                nvidia-docker exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; wget https://s3.amazonaws.com/h2o-public-test-data/h2o4gpu/open_data/kmeans_data/h2o-logo.jpg'
                                nvidia-docker exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; cp /data/ipums_1k.csv .'
                                nvidia-docker exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; cp /data/ipums.feather .'
                                nvidia-docker save opsh2oai/h2o4gpu$-${versionTag}{extratag}-runtime > h2o4gpu-${versionTag}${extratag}-runtime.tar
                                gzip  h2o4gpu-${versionTag}${extratag}-runtime.tar
                                nvidia-docker stop ${CONTAINER_NAME}
                            """
                        stash includes: 'h2o4gpu-${versionTag}${extratag}-runtime.tar.gz', name: 'docker-${versionTag}${extratag}-runtime'
                        // Archive artifacts
                        arch 'h2o4gpu-${versionTag}${extratag}-runtime.tar.gz'
                    }
                }
            }
        }

        stage('Publish Runtime Docker for nccl CUDA8 to S3') {
            agent {
                label "linux"
            }

            steps {
                unstash 'version_info'
                def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                def extratag = "_nccl_cuda8"
                unstash 'docker-${versionTag}${extratag}-runtime'
                sh 'echo "Stashed files:" && ls -l docker*'
                def artifactId = h2o4gpu
                def artifact = "${artifactId}-${versionTag}${extratag}-runtime.tar.gz"
                def localArtifact = "${artifact}"

                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                    script {
                        s3up_simple(${versionTag}, ${extratag}, ${artifactId}, ${artifact}, ${localArtifact})
                    }
                    }
                }
            }
        }











        stage('Build on Linux nonccl CUDA8') {
            agent {
                label "nvidia-docker && (mr-dl11||mr-dl16||mr-dl10)"
            }

            steps {
                dumpInfo 'Linux Build Info'
                // Do checkout
                retryWithTimeout(100 /* seconds */, 3 /* retries */) {
                    deleteDir()
                    checkout([
                            $class                           : 'GitSCM',
                            branches                         : scm.branches,
                            doGenerateSubmoduleConfigurations: false,
                            extensions                       : scm.extensions + [[$class: 'SubmoduleOption', disableSubmodules: true, recursiveSubmodules: false, reference: '', trackingSubmodules: false, shallow: true]],
                            submoduleCfg                     : [],
                            userRemoteConfigs                : scm.userRemoteConfigs])
                }

                script {
                    def extratag = "_nonccl_cuda8"
                    CONTAINER_NAME = "h2o4gpu-${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    echo "CONTAINER_NAME = ${CONTAINER_NAME}"
                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        try {
                            sh """
                                    nvidia-docker build  -t opsh2oai/h2o4gpu-${extratag}-build -f Dockerfile-build  --build-arg cuda=nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04 .
                                    nvidia-docker run --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-${extratag}-build
                                    nvidia-docker exec ${CONTAINER_NAME} rm -rf data
                                    nvidia-docker exec ${CONTAINER_NAME} ln -s /data ./data
                                    nvidia-docker exec ${CONTAINER_NAME} rm -rf open_data
                                    nvidia-docker exec ${CONTAINER_NAME} ln -s /open_data ./open_data
                                    nvidia-docker exec ${CONTAINER_NAME} bash -c 'eval \"\$(/root/.pyenv/bin/pyenv init -)\" ; /root/.pyenv/bin/pyenv global 3.6.1; ./scripts/gitshallow_submodules.sh; make ${env.MAKE_OPTS} AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} fullinstalljenkins${extratag} ; rm -rf build/VERSION.txt ; make build/VERSION.txt'
                                """
                            stash includes: 'src/interface_py/dist2/*.whl', name: 'linux_whl2'
                            stash includes: 'build/VERSION.txt', name: 'version_info'
                            // Archive artifacts
                            arch 'src/interface_py/dist2/*.whl'
                        } finally {
                            sh "nvidia-docker stop ${CONTAINER_NAME}"
                        }
                    }
                }
            }
        }
        stage('Publish to S3 nonccl CUDA8') {
            agent {
                label "linux"
            }

            steps {
                unstash 'version_info'
                def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                def extratag = "_nonccl_cuda8"
                unstash 'linux_whl2'
                sh 'echo "Stashed files:" && ls -l src/interface_py/dist2/'
                def artifactId = h2o4gpu
                def artifact = "${artifactId}-${versionTag}-py36-none-any.whl"
                def localArtifact = "src/interface_py/dist2/${artifact}"

                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                script {
                    s3up_simple(${versionTag}, ${extratag}, ${artifactId}, ${artifact}, ${localArtifact})
                }
                }
            }
        }
        stage('Build Runtime Docker for nonccl CUDA8') {
            agent {
                label "nvidia-docker && (mr-dl11||mr-dl16||mr-dl10)"
            }

            steps {
                dumpInfo 'Linux Build Info'
                // Do checkout
                retryWithTimeout(100 /* seconds */, 3 /* retries */) {
                    deleteDir()
                    checkout([
                            $class                           : 'GitSCM',
                            branches                         : scm.branches,
                            doGenerateSubmoduleConfigurations: false,
                            extensions                       : scm.extensions + [[$class: 'SubmoduleOption', disableSubmodules: true, recursiveSubmodules: false, reference: '', trackingSubmodules: false, shallow: true]],
                            submoduleCfg                     : [],
                            userRemoteConfigs                : scm.userRemoteConfigs])
                }
                script {
                    sh """
                        mkdir -p build ; rm -rf build/VERSION.txt
                    """
                }
                unstash 'version_info'
                sh 'echo "Stashed version file:" && ls -l build/'

                script {
                    def extratag = "_nonccl_cuda8"
                    def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                    CONTAINER_NAME = "h2o4gpu-runtime-${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    echo "CONTAINER_NAME = ${CONTAINER_NAME}"
                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        sh """
                                nvidia-docker build  -t opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime:latest -f Dockerfile-runtime --build-arg cuda=nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04 --build-arg wheel=${versionTag}${extratag}/h2o4gpu-${versionTag}-py36-none-any.whl .
                                nvidia-docker run  --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime
                                nvidia-docker exec ${CONTAINER_NAME} rm -rf data
                                nvidia-docker exec ${CONTAINER_NAME} ln -s /data ./data
                                nvidia-docker exec ${CONTAINER_NAME} rm -rf open_data
                                nvidia-docker exec ${CONTAINER_NAME} ln -s /open_data ./open_data
                                nvidia-docker exec ${CONTAINER_NAME} bash -c '. /h2o4gpu_env/bin/activate ; pip freeze'
                                nvidia-docker exec ${CONTAINER_NAME} bash -c '. /h2o4gpu_env/bin/activate ; cd /jupyter/demos ; python -c "exec(\\"from sklearn.datasets import fetch_covtype\\ncov = fetch_covtype()\\")"'
                                nvidia-docker exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; cp /open_data/creditcard.csv .'
                                nvidia-docker exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; wget https://s3.amazonaws.com/h2o-public-test-data/h2o4gpu/open_data/kmeans_data/h2o-logo.jpg'
                                nvidia-docker exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; cp /data/ipums_1k.csv .'
                                nvidia-docker exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; cp /data/ipums.feather .'
                                nvidia-docker save opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime > h2o4gpu-${versionTag}${extratag}-runtime.tar
                                gzip  h2o4gpu-${versionTag}${extratag}-runtime.tar
                            """
                        stash includes: 'h2o4gpu-${versionTag}${extratag}-runtime.tar.gz', name: 'docker-${versionTag}${extratag}-runtime'
                        // Archive artifacts
                        arch 'h2o4gpu-${versionTag}${extratag}-runtime.tar.gz'
                    }
                }
            }
        }

        stage('Publish Runtime Docker for nonccl CUDA8 to S3') {
            agent {
                label "linux"
            }

            steps {
                unstash 'version_info'
                def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                def extratag = "_nonccl_cuda8"
                unstash 'docker${versionTag}${extratag}-runtime'
                sh 'echo "Stashed files:" && ls -l docker*'
                def artifactId = h2o4gpu
                def artifact = "${artifactId}-${versionTag}${extratag}-runtime.tar.gz"
                def localArtifact = "${artifact}"

                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                    script {
                        s3up_simple(${versionTag}, ${extratag}, ${artifactId}, ${artifact}, ${localArtifact})
                    }
                    }
                }
            }
        }










        stage('Build on Linux nccl CUDA9') {
            agent {
                label "nvidia-docker && (mr-dl11||mr-dl16||mr-dl10)"
            }

            steps {
                dumpInfo 'Linux Build Info'
                // Do checkout
                retryWithTimeout(100 /* seconds */, 3 /* retries */) {
                    deleteDir()
                    checkout([
                            $class                           : 'GitSCM',
                            branches                         : scm.branches,
                            doGenerateSubmoduleConfigurations: false,
                            extensions                       : scm.extensions + [[$class: 'SubmoduleOption', disableSubmodules: true, recursiveSubmodules: false, reference: '', trackingSubmodules: false, shallow: true]],
                            submoduleCfg                     : [],
                            userRemoteConfigs                : scm.userRemoteConfigs])
                }

                script {
                    def extratag = "_nccl_cuda9"
                    CONTAINER_NAME = "h2o4gpu-${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    echo "CONTAINER_NAME = ${CONTAINER_NAME}"
                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        try {
                            sh """
                                    nvidia-docker build  -t opsh2oai/h2o4gpu-${extratag}-build -f Dockerfile-build --build-arg cuda=nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 .
                                    nvidia-docker run --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-${extratag}-build
                                    nvidia-docker exec ${CONTAINER_NAME} rm -rf data
                                    nvidia-docker exec ${CONTAINER_NAME} ln -s /data ./data
                                    nvidia-docker exec ${CONTAINER_NAME} rm -rf open_data
                                    nvidia-docker exec ${CONTAINER_NAME} ln -s /open_data ./open_data
                                    nvidia-docker exec ${CONTAINER_NAME} bash -c 'eval \"\$(/root/.pyenv/bin/pyenv init -)\" ; /root/.pyenv/bin/pyenv global 3.6.1; ./scripts/gitshallow_submodules.sh; make ${env.MAKE_OPTS} AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} fullinstalljenkins${extratag} ; rm -rf build/VERSION.txt ; make build/VERSION.txt'
                                """
                            stash includes: 'src/interface_py/dist4/*.whl', name: 'linux_whl4'
                            stash includes: 'build/VERSION.txt', name: 'version_info'
                            // Archive artifacts
                            arch 'src/interface_py/dist4/*.whl'
                        } finally {
                            sh "nvidia-docker stop ${CONTAINER_NAME}"
                        }
                    }
                }
            }
        }
        stage('Publish to S3 nccl CUDA9') {
            agent {
                label "linux"
            }

            steps {
                unstash 'version_info'
                def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                unstash 'linux_whl4'
                sh 'echo "Stashed files:" && ls -l src/interface_py/dist4/'
                def extratag = "_nccl_cuda9"
                def artifactId = h2o4gpu
                def artifact = "${artifactId}-${versionTag}-py36-none-any.whl"
                def localArtifact = "src/interface_py/dist4/${artifact}"

                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                script {
                    s3up_simple(${versionTag}, ${extratag}, ${artifactId}, ${artifact}, ${localArtifact})
                }
                }
            }
        }

        stage('Build Runtime Docker for nccl CUDA9') {
            agent {
                label "nvidia-docker && (mr-dl11||mr-dl16||mr-dl10)"
            }

            steps {
                dumpInfo 'Linux Build Info'
                // Do checkout
                retryWithTimeout(100 /* seconds */, 3 /* retries */) {
                    deleteDir()
                    checkout([
                            $class                           : 'GitSCM',
                            branches                         : scm.branches,
                            doGenerateSubmoduleConfigurations: false,
                            extensions                       : scm.extensions + [[$class: 'SubmoduleOption', disableSubmodules: true, recursiveSubmodules: false, reference: '', trackingSubmodules: false, shallow: true]],
                            submoduleCfg                     : [],
                            userRemoteConfigs                : scm.userRemoteConfigs])
                }
                script {
                    sh """
                        mkdir -p build ; rm -rf build/VERSION.txt
                    """
                }
                unstash 'version_info'
                sh 'echo "Stashed version file:" && ls -l build/'

                script {
                    def extratag = "_nccl_cuda9"
                    def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                    CONTAINER_NAME = "h2o4gpu-runtime-${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        sh """
                                nvidia-docker build  -t opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime:latest -f Dockerfile-runtime --build-arg cuda=nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04 --build-arg wheel=${versionTag}_nccl_cuda9/h2o4gpu-${versionTag}-py36-none-any.whl .
                                nvidia-docker run  --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime
                                nvidia-docker exec ${CONTAINER_NAME} rm -rf data
                                nvidia-docker exec ${CONTAINER_NAME} ln -s /data ./data
                                nvidia-docker exec ${CONTAINER_NAME} rm -rf open_data
                                nvidia-docker exec ${CONTAINER_NAME} ln -s /open_data ./open_data
                                nvidia-docker exec ${CONTAINER_NAME} bash -c '. /h2o4gpu_env/bin/activate ; pip freeze'
                                nvidia-docker exec ${CONTAINER_NAME} bash -c '. /h2o4gpu_env/bin/activate ; cd /jupyter/demos ; python -c "exec(\\"from sklearn.datasets import fetch_covtype\\ncov = fetch_covtype()\\")"'
                                nvidia-docker exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; cp /open_data/creditcard.csv .'
                                nvidia-docker exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; wget https://s3.amazonaws.com/h2o-public-test-data/h2o4gpu/open_data/kmeans_data/h2o-logo.jpg'
                                nvidia-docker exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; /data/ipums_1k.csv .'
                                nvidia-docker exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; /data/ipums.feather .'
                                nvidia-docker save opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime > h2o4gpu-${versionTag}${extratag}-runtime.tar
                                gzip  h2o4gpu-${versionTag}${extratag}-runtime.tar
                            """
                        stash includes: 'h2o4gpu-${versionTag}${extratag}-runtime.tar.gz', name: 'docker-${versionTag}${extratag}-runtime'
                        // Archive artifacts
                        arch 'h2o4gpu-${versionTag}${extratag}-runtime.tar.gz'
                    }
                }
            }
        }

        stage('Publish Runtime Docker for nccl CUDA9 to S3') {
            agent {
                label "linux"
            }

            steps {
                unstash 'version_info'
                def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                def extratag = "_nccl_cuda9"
                unstash 'docker-${versionTag}${extratag}-runtime'
                sh 'echo "Stashed files:" && ls -l docker*'
                def artifactId = h2o4gpu
                def artifact = "${artifactId}-${versionTag}${extratag}-runtime.tar.gz"
                def localArtifact = "${artifact}"

                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                    script {
                        s3up_simple(${versionTag}, ${extratag}, ${artifactId}, ${artifact}, ${localArtifact})
                    }
                    }
                }
            }
        }







        stage('Build on Linux nonccl CUDA9') {
            agent {
                label "nvidia-docker && (mr-dl11||mr-dl16||mr-dl10)"
            }

            steps {
                dumpInfo 'Linux Build Info'
                // Do checkout
                retryWithTimeout(100 /* seconds */, 3 /* retries */) {
                    deleteDir()
                    checkout([
                            $class                           : 'GitSCM',
                            branches                         : scm.branches,
                            doGenerateSubmoduleConfigurations: false,
                            extensions                       : scm.extensions + [[$class: 'SubmoduleOption', disableSubmodules: true, recursiveSubmodules: false, reference: '', trackingSubmodules: false, shallow: true]],
                            submoduleCfg                     : [],
                            userRemoteConfigs                : scm.userRemoteConfigs])
                }
                script {
                    def extratag = "_nonccl_cuda9"
                    CONTAINER_NAME = "h2o4gpu-${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    echo "CONTAINER_NAME = ${CONTAINER_NAME}"
                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        try {
                            sh """
                                    nvidia-docker build  -t opsh2oai/h2o4gpu-${extratag}-build -f Dockerfile-build --build-arg cuda=nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 .
                                    nvidia-docker run --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-${extratag}-build
                                    nvidia-docker exec ${CONTAINER_NAME} rm -rf data
                                    nvidia-docker exec ${CONTAINER_NAME} ln -s /data ./data
                                    nvidia-docker exec ${CONTAINER_NAME} rm -rf open_data
                                    nvidia-docker exec ${CONTAINER_NAME} ln -s /open_data ./open_data
                                    nvidia-docker exec ${CONTAINER_NAME} bash -c 'eval \"\$(/root/.pyenv/bin/pyenv init -)\" ; /root/.pyenv/bin/pyenv global 3.6.1; ./scripts/gitshallow_submodules.sh; make ${env.MAKE_OPTS} AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} fullinstalljenkins_nonccl_cuda9 ; rm -rf build/VERSION.txt ; make build/VERSION.txt'
                                """
                            stash includes: 'src/interface_py/dist3/*.whl', name: 'linux_whl3'
                            stash includes: 'build/VERSION.txt', name: 'version_info'
                            // Archive artifacts
                            arch 'src/interface_py/dist3/*.whl'
                        } finally {
                            sh "nvidia-docker stop ${CONTAINER_NAME}"
                        }
                    }
                }
            }
        }
        stage('Publish to S3 nonccl CUDA9') {
            agent {
                label "linux"
            }

            steps {
                unstash 'version_info'
                def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                def extratag = "_nonccl_cuda9"
                unstash 'linux_whl3'
                sh 'echo "Stashed files:" && ls -l src/interface_py/dist3/'
                def artifactId = h2o4gpu
                def artifact = "${artifactId}-${versionTag}-py36-none-any.whl"
                def localArtifact = "src/interface_py/dist3/${artifact}"

                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                    script {
                        s3up_simple(${versionTag}, ${extratag}, ${artifactId}, ${artifact}, ${localArtifact})
                    }
                }
                }
            }
        }

        stage('Build Runtime Docker for nonccl CUDA9') {
            agent {
                label "nvidia-docker && (mr-dl11||mr-dl16||mr-dl10)"
            }

            steps {
                dumpInfo 'Linux Build Info'
                // Do checkout
                retryWithTimeout(100 /* seconds */, 3 /* retries */) {
                    deleteDir()
                    checkout([
                            $class                           : 'GitSCM',
                            branches                         : scm.branches,
                            doGenerateSubmoduleConfigurations: false,
                            extensions                       : scm.extensions + [[$class: 'SubmoduleOption', disableSubmodules: true, recursiveSubmodules: false, reference: '', trackingSubmodules: false, shallow: true]],
                            submoduleCfg                     : [],
                            userRemoteConfigs                : scm.userRemoteConfigs])
                }
                script {
                    sh """
                        mkdir -p build ; rm -rf build/VERSION.txt
                    """
                }
                unstash 'version_info'
                sh 'echo "Stashed version file:" && ls -l build/'

                script {
                    def extratag = "_nonccl_cuda9"
                    def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                    CONTAINER_NAME = "h2o4gpu-runtime-${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    echo "CONTAINER_NAME = ${CONTAINER_NAME}"
                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        sh """
                                nvidia-docker build  -t opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime:latest -f Dockerfile-runtime --build-arg cuda=nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04  --build-arg wheel=${versionTag}${extratag}/h2o4gpu-${versionTag}-py36-none-any.whl .
                                nvidia-docker run  --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime
                                nvidia-docker exec ${CONTAINER_NAME} rm -rf data
                                nvidia-docker exec ${CONTAINER_NAME} ln -s /data ./data
                                nvidia-docker exec ${CONTAINER_NAME} rm -rf open_data
                                nvidia-docker exec ${CONTAINER_NAME} ln -s /open_data ./open_data
                                nvidia-docker exec ${CONTAINER_NAME} bash -c '. /h2o4gpu_env/bin/activate ; pip freeze'
                                nvidia-docker exec ${CONTAINER_NAME} bash -c '. /h2o4gpu_env/bin/activate ; cd /jupyter/demos ; python -c "exec(\\"from sklearn.datasets import fetch_covtype\\ncov = fetch_covtype()\\")"'
                                nvidia-docker exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; cp /open_data/creditcard.csv .'
                                nvidia-docker exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; wget https://s3.amazonaws.com/h2o-public-test-data/h2o4gpu/open_data/kmeans_data/h2o-logo.jpg'
                                nvidia-docker exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; cp /data/ipums_1k.csv .'
                                nvidia-docker exec ${CONTAINER_NAME} bash -c 'cd /jupyter/demos ; cp /data/ipums.feather .'
                                nvidia-docker save opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime > h2o4gpu-${versionTag}${extratag}-runtime.tar
                                gzip h2o4gpu-${versionTag}${extratag}-runtime.tar
                            """
                        stash includes: 'h2o4gpu-${versionTag}${extratag}-runtime.tar.gz', name: 'docker-${versionTag}${extratag}-runtime'
                        // Archive artifacts
                        arch 'h2o4gpu-${versionTag}${extratag}-runtime.tar.gz'
                    }
                }
            }
        }

        stage('Publish Runtime Docker for nonccl CUDA9 to S3') {
            agent {
                label "linux"
            }

            steps {
                unstash 'version_info'
                def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                def extratag = "_nonccl_cuda9"
                unstash 'docker-${versionTag}${extratag}-runtime'
                sh 'echo "Stashed files:" && ls -l docker*'
                def artifactId = h2o4gpu
                def artifact = "${artifactId}-${versionTag}${extratag}-runtime.tar.gz"
                def localArtifact = "${artifact}"

                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                    script {
                        s3up_simple(${versionTag}, ${extratag}, ${artifactId}, ${artifact}, ${localArtifact})
                    }
                    }
                }


        }











    }
    post {
        failure {
            node('mr-dl11') {
                script {
                    // Hack - the email plugin finds 0 recipients for the first commit of each new PR build...
                    def email = utilsLib.getCommandOutput("git --no-pager show -s --format='%ae'")
                    emailext(
                            to: "mateusz@h2o.ai, ${email}",
                            subject: "BUILD FAILED: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]'",
                            body: '''${JELLY_SCRIPT, template="html_gmail"}''',
                            attachLog: true,
                            compressLog: true,
                            recipientProviders: [
                                    [$class: 'CulpritsRecipientProvider'],
                                    [$class: 'DevelopersRecipientProvider'],
                                    [$class: 'FailingTestSuspectsRecipientProvider'],
                                    [$class: 'FirstFailingBuildSuspectsRecipientProvider'],
                                    [$class: 'RequesterRecipientProvider']
                            ]
                    )
                }
            }
        }
    }
}


def s3up_simple(versionTag, extratag, artifactId, artifact, localArtifact) {
    if (isRelease()) {
        def bucket = "s3://artifacts.h2o.ai/releases/stable/ai/h2o/${artifactId}/${versionTag}${extratag}/"
        sh "s3cmd put ${localArtifact} ${bucket}"
        sh "s3cmd setacl --acl-public  ${bucket}${artifact}"
    }
    if (isBleedingEdge()) {
        def bucket = "s3://artifacts.h2o.ai/releases/bleeding-edge/ai/h2o/${artifactId}/${versionTag}${extratag}/"
        sh "s3cmd put ${localArtifact} ${bucket}"
        sh "s3cmd setacl --acl-public  ${bucket}${artifact}"
    }
    if (!(isRelease() || isBleedingEdge())) {
        // always upload for testing
        def bucket = "s3://artifacts.h2o.ai/releases/snapshots_other/ai/h2o/${artifactId}/${versionTag}${extratag}/"
        sh "s3cmd put ${localArtifact} ${bucket}"
        sh "s3cmd setacl --acl-public  ${bucket}${artifact}"
    }
}

def isRelease() {
    return env.BRANCH_NAME.startsWith("rel")
}

def isBleedingEdge() {
    return env.BRANCH_NAME.startsWith("master")
}

