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

        stage('Build on Linux CUDA8') {
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
                    CONTAINER_NAME = "h2o4gpu-${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        try {
                            sh """
                                    nvidia-docker build  -t opsh2oai/h2o4gpu-build -f Dockerfile-build --build-arg cuda=nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04 .
                                    nvidia-docker run --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-build
                                    nvidia-docker exec ${CONTAINER_NAME} rm -rf data
                                    nvidia-docker exec ${CONTAINER_NAME} ln -s /data ./data
                                    nvidia-docker exec ${CONTAINER_NAME} rm -rf open_data
                                    nvidia-docker exec ${CONTAINER_NAME} ln -s /open_data ./open_data
                                    nvidia-docker exec ${
                                CONTAINER_NAME
                            } bash -c 'eval \"\$(/root/.pyenv/bin/pyenv init -)\" ; /root/.pyenv/bin/pyenv global 3.6.1; ./scripts/gitshallow_submodules.sh; make ${
                                env.MAKE_OPTS
                            } AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} fullinstalljenkins_nccl_cuda8 ; rm -rf build/VERSION.txt ; make build/VERSION.txt'
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

        stage('Full Test on Linux CUDA8') {
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
                script {
                    try {
                        sh """
                            nvidia-docker run  --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-build
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

        stage('Pylint on Linux CUDA8') {
            agent {
                label "gpu && nvidia-docker && (mr-dl11||mr-dl16||mr-dl10)"
            }

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
                            nvidia-docker build  -t opsh2oai/h2o4gpu-build -f Dockerfile-build  --build-arg cuda=nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04 .
                            nvidia-docker run  --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-build
                            nvidia-docker exec ${CONTAINER_NAME} touch src/interface_py/h2o4gpu/__init__.py
                            nvidia-docker exec ${CONTAINER_NAME} bash -c 'eval \"\$(/root/.pyenv/bin/pyenv init -)\"  ;  /root/.pyenv/bin/pyenv global 3.6.1; make pylint'
                            nvidia-docker stop ${CONTAINER_NAME}
                        """
                }
            }
        }

        stage('Publish to S3 CUDA8') {
            agent {
                label "linux"
            }

            steps {
                unstash 'linux_whl'
                unstash 'version_info'
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                    sh 'echo "Stashed files:" && ls -l src/interface_py/dist/'
                    script {
                        // Load the version file content
                        def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                        def version = utilsLib.fragmentVersion(versionTag)
                        def _majorVersion = version[0]
                        def _buildVersion = version[1]
                        version = null // This is necessary, else version:Tuple will be serialized

                        if (isRelease()) {
                            s3up {
                                localArtifact = 'src/interface_py/dist/h2o4gpu-*-py36-none-any.whl'
                                artifactId = "h2o4gpu"
                                majorVersion = _majorVersion
                                buildVersion = _buildVersion
                                keepPrivate = false
                                remoteArtifactBucket = "s3://artifacts.h2o.ai/releases/stable"
                            }
                            sh "s3cmd setacl --acl-public s3://artifacts.h2o.ai/releases/stable/ai/h2o/h2o4gpu/${versionTag}/h2o4gpu-${versionTag}-py36-none-any.whl"
                        }

                        if (isBleedingEdge()) {
                            s3up {
                                localArtifact = 'src/interface_py/dist/h2o4gpu-*-py36-none-any.whl'
                                artifactId = "h2o4gpu"
                                majorVersion = _majorVersion
                                buildVersion = _buildVersion
                                keepPrivate = false
                                remoteArtifactBucket = "s3://artifacts.h2o.ai/releases/bleeding-edge"
                            }
                            sh "s3cmd setacl --acl-public s3://artifacts.h2o.ai/releases/bleeding-edge/ai/h2o/h2o4gpu/${versionTag}/h2o4gpu-${versionTag}-py36-none-any.whl"
                        }
                    }
                    }
                }
            }
        }

        stage('Build Runtime Docker for CUDA8') {
            agent {
                label "nvidia-docker && (mr-dl11||mr-dl16||mr-dl10)"
            }

            steps {
                dumpInfo 'Linux Build Info'
                unstash 'version_info'
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
                        rm -rf build/VERSION.txt ; make build/VERSION.txt
                    """
                    def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                    CONTAINER_NAME = "h2o4gpu-cuda8-runtime${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        sh """
                                nvidia-docker build -t opsh2oai/h2o4gpu-cuda8-runtime:latest -f Dockerfile-runtime --build-arg cuda=nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04 --build-arg wheel=${versionTag}/h2o4gpu-${versionTag}-py36-none-any.whl .
                                nvidia-docker run  --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-cuda8-runtime
                                nvidia-docker exec ${CONTAINER_NAME} rm -rf data
                                nvidia-docker exec ${CONTAINER_NAME} ln -s /data ./data
                                nvidia-docker exec ${CONTAINER_NAME} rm -rf open_data
                                nvidia-docker exec ${CONTAINER_NAME} ln -s /open_data ./open_data
                                nvidia-docker exec ${CONTAINER_NAME} cd /jupyter/demos ; python -c "exec(\"from sklearn.datasets import fetch_covtype\\ncov = fetch_covtype()\")"
                                nvidia-docker exec ${CONTAINER_NAME} cd /jupyter/demos ; cp /open_data/creditcard.csv .
                                nvidia-docker exec ${CONTAINER_NAME} cd /jupyter/demos ; wget https://s3.amazonaws.com/h2o-public-test-data/h2o4gpu/open_data/kmeans_data/h2o-logo.jpg
                                nvidia-docker exec ${CONTAINER_NAME} cd /jupyter/demos ; cp /home/0xdiag/h2o4gpu/data/ipums_1k.csv .
                                nvidia-docker exec ${CONTAINER_NAME} cd /jupyter/demos ; cp /home/0xdiag/h2o4gpu/data/ipums.feather .
                                nvidia-docker save opsh2oai/h2o4gpu-cuda8-runtime > h2o4gpu-cuda8-runtime.tar
                                gzip  h2o4gpu-cuda8-runtime.tar
                                nvidia-docker stop ${CONTAINER_NAME}
                            """
                        stash includes: 'h2o4gpu-cuda8-runtime.tar.gz', name: 'docker-cuda8-runtime'
                        // Archive artifacts
                        arch 'h2o4gpu-cuda8-runtime.tar.gz'
                    }
                }
            }
        }

        stage('Publish Runtime Docker for CUDA8 to S3') {
            agent {
                label "linux"
            }

            steps {
                unstash 'docker-cuda8-runtime'
                unstash 'version_info'
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                    script {
                        // Load the version file content
                        def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                        def version = utilsLib.fragmentVersion(versionTag)
                        def _majorVersion = version[0]
                        def _buildVersion = version[1]
                        version = null // This is necessary, else version:Tuple will be serialized

                        if (isRelease()) {
                            s3up {
                                localArtifact = 'h2o4gpu-cuda8-runtime.tar.gz'
                                artifactId = "h2o4gpu"
                                majorVersion = _majorVersion
                                buildVersion = _buildVersion
                                keepPrivate = false
                                remoteArtifactBucket = "s3://artifacts.h2o.ai/releases/stable"
                            }
                            sh "s3cmd setacl --acl-public s3://artifacts.h2o.ai/releases/stable/ai/h2o/h2o4gpu/${versionTag}/h2o4gpu-cuda8-runtime.tar.gz"
                        }

                        if (isBleedingEdge()) {
                            s3up {
                                localArtifact = 'h2o4gpu-cuda8-runtime.tar.gz'
                                artifactId = "h2o4gpu"
                                majorVersion = _majorVersion
                                buildVersion = _buildVersion
                                keepPrivate = false
                                remoteArtifactBucket = "s3://artifacts.h2o.ai/releases/bleeding-edge"
                            }
                            sh "s3cmd setacl --acl-public s3://artifacts.h2o.ai/releases/bleeding-edge/ai/h2o/h2o4gpu/${versionTag}/h2o4gpu-cuda8-runtime.tar.gz"
                        }
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
                    CONTAINER_NAME = "h2o4gpu-nonccl-cuda8-${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        try {
                            sh """
                                    nvidia-docker build  -t opsh2oai/h2o4gpu-build -f Dockerfile-build  --build-arg cuda=nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04 .
                                    nvidia-docker run --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-build
                                    nvidia-docker exec ${CONTAINER_NAME} rm -rf data
                                    nvidia-docker exec ${CONTAINER_NAME} ln -s /data ./data
                                    nvidia-docker exec ${CONTAINER_NAME} rm -rf open_data
                                    nvidia-docker exec ${CONTAINER_NAME} ln -s /open_data ./open_data
                                    nvidia-docker exec ${CONTAINER_NAME} bash -c 'eval \"\$(/root/.pyenv/bin/pyenv init -)\" ; /root/.pyenv/bin/pyenv global 3.6.1; ./scripts/gitshallow_submodules.sh; make ${env.MAKE_OPTS} AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} fullinstalljenkins_nonccl_cuda8 ; rm -rf build/VERSION.txt ; make build/VERSION.txt'
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
                unstash 'linux_whl2'
                unstash 'version_info'
                sh 'echo "Stashed files:" && ls -l src/interface_py/dist2/'
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                script {
                    // Load the version file content
                    def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                    def version = utilsLib.fragmentVersion(versionTag)
                    def _majorVersion = version[0]
                    def _buildVersion = version[1]
                    version = null // This is necessary, else version:Tuple will be serialized

                    if (isRelease()) {
                        def artifact = "h2o4gpu-${versionTag}-py36-none-any.whl"
                        def localArtifact = "src/interface_py/dist2/${artifact}"
                        def bucket = "s3://artifacts.h2o.ai/releases/stable/ai/h2o/h2o4gpu/${versionTag}_nonccl_cuda8/"
                        sh "s3cmd put ${localArtifact} ${bucket}"
                        sh "s3cmd setacl --acl-public  ${bucket}${artifact}"
                    }

                    if (isBleedingEdge()) {
                        def artifact = "h2o4gpu-${versionTag}-py36-none-any.whl"
                        def localArtifact = "src/interface_py/dist2/${artifact}"
                        def bucket = "s3://artifacts.h2o.ai/releases/bleeding-edge/ai/h2o/h2o4gpu/${versionTag}_nonccl_cuda8/"
                        sh "s3cmd put ${localArtifact} ${bucket}"
                        sh "s3cmd setacl --acl-public  ${bucket}${artifact}"
                    }
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
                        rm -rf build/VERSION.txt ; make build/VERSION.txt
                    """
                    def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                    CONTAINER_NAME = "h2o4gpu-nonccl-cuda8-runtime-${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        sh """
                                nvidia-docker build  -t opsh2oai/h2o4gpu-nonccl-cuda8-runtime:latest -f Dockerfile-runtime --build-arg cuda=nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04 --build-arg wheel=${versionTag}_nonccl_cuda8/h2o4gpu-${versionTag}-py36-none-any.whl .
                                nvidia-docker run  --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-nonccl-cuda8-runtime
                                nvidia-docker exec ${CONTAINER_NAME} rm -rf data
                                nvidia-docker exec ${CONTAINER_NAME} ln -s /data ./data
                                nvidia-docker exec ${CONTAINER_NAME} rm -rf open_data
                                nvidia-docker exec ${CONTAINER_NAME} ln -s /open_data ./open_data
                                nvidia-docker exec ${CONTAINER_NAME} cd /jupyter/demos ; python -c "exec(\"from sklearn.datasets import fetch_covtype\\ncov = fetch_covtype()\")"
                                nvidia-docker exec ${CONTAINER_NAME} cd /jupyter/demos ; cp /open_data/creditcard.csv .
                                nvidia-docker exec ${CONTAINER_NAME} cd /jupyter/demos ; wget https://s3.amazonaws.com/h2o-public-test-data/h2o4gpu/open_data/kmeans_data/h2o-logo.jpg
                                nvidia-docker exec ${CONTAINER_NAME} cd /jupyter/demos ; cp /home/0xdiag/h2o4gpu/data/ipums_1k.csv .
                                nvidia-docker exec ${CONTAINER_NAME} cd /jupyter/demos ; cp /home/0xdiag/h2o4gpu/data/ipums.feather .
                                nvidia-docker save opsh2oai/h2o4gpu-nonccl-cuda8-runtime > h2o4gpu-nonccl-cuda8-runtime.tar
                                gzip  h2o4gpu-nonccl-cuda8-runtime.tar
                            """
                        stash includes: 'h2o4gpu-nonccl-cuda8-runtime.tar.gz', name: 'docker-nonccl-cuda8-runtime'
                        // Archive artifacts
                        arch 'h2o4gpu-nonccl-cuda8-runtime.tar.gz'
                    }
                }
            }
        }

        stage('Publish Runtime Docker for nonccl CUDA8 to S3') {
            agent {
                label "linux"
            }

            steps {
                unstash 'docker-cuda8-runtime'
                unstash 'version_info'
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                    script {
                        // Load the version file content
                        def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                        def version = utilsLib.fragmentVersion(versionTag)
                        def _majorVersion = version[0]
                        def _buildVersion = version[1]
                        version = null // This is necessary, else version:Tuple will be serialized

                        if (isRelease()) {
                            s3up {
                                localArtifact = 'h2o4gpu-nonccl-cuda8-runtime.tar.gz'
                                artifactId = "h2o4gpu"
                                majorVersion = _majorVersion
                                buildVersion = _buildVersion
                                keepPrivate = false
                                remoteArtifactBucket = "s3://artifacts.h2o.ai/releases/stable"
                            }
                            sh "s3cmd setacl --acl-public s3://artifacts.h2o.ai/releases/stable/ai/h2o/h2o4gpu/${versionTag}_nonccl_cuda8/h2o4gpu-nonccl-cuda8-runtime.tar.gz"
                        }

                        if (isBleedingEdge()) {
                            s3up {
                                localArtifact = 'h2o4gpu-nonccl-cuda8-runtime.tar.gz'
                                artifactId = "h2o4gpu"
                                majorVersion = _majorVersion
                                buildVersion = _buildVersion
                                keepPrivate = false
                                remoteArtifactBucket = "s3://artifacts.h2o.ai/releases/bleeding-edge"
                            }
                            sh "s3cmd setacl --acl-public s3://artifacts.h2o.ai/releases/bleeding-edge/ai/h2o/h2o4gpu/${versionTag}_nonccl_cuda8/h2o4gpu-nonccl-cuda8-runtime.tar.gz"
                        }
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
                    CONTAINER_NAME = "h2o4gpu-nccl-cuda9-${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        try {
                            sh """
                                    nvidia-docker build  -t opsh2oai/h2o4gpu-build -f Dockerfile-build --build-arg cuda=nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 .
                                    nvidia-docker run --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-build
                                    nvidia-docker exec ${CONTAINER_NAME} rm -rf data
                                    nvidia-docker exec ${CONTAINER_NAME} ln -s /data ./data
                                    nvidia-docker exec ${CONTAINER_NAME} rm -rf open_data
                                    nvidia-docker exec ${CONTAINER_NAME} ln -s /open_data ./open_data
                                    nvidia-docker exec ${CONTAINER_NAME} bash -c 'eval \"\$(/root/.pyenv/bin/pyenv init -)\" ; /root/.pyenv/bin/pyenv global 3.6.1; ./scripts/gitshallow_submodules.sh; make ${env.MAKE_OPTS} AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} fullinstalljenkins_nccl_cuda9 ; rm -rf build/VERSION.txt ; make build/VERSION.txt'
                                """
                            stash includes: 'src/interface_py/dist4/*.whl', name: 'linux_whl3'
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
                unstash 'linux_whl3'
                unstash 'version_info'
                sh 'echo "Stashed files:" && ls -l src/interface_py/dist4/'
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                script {
                    // Load the version file content
                    def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                    def version = utilsLib.fragmentVersion(versionTag)
                    def _majorVersion = version[0]
                    def _buildVersion = version[1]
                    version = null // This is necessary, else version:Tuple will be serialized

                    if (isRelease()) {
                        def artifact = "h2o4gpu-${versionTag}-py36-none-any.whl"
                        def localArtifact = "src/interface_py/dist4/${artifact}"
                        def bucket = "s3://artifacts.h2o.ai/releases/stable/ai/h2o/h2o4gpu/${versionTag}_nccl_cuda9/"
                        sh "s3cmd put ${localArtifact} ${bucket}"
                        sh "s3cmd setacl --acl-public  ${bucket}${artifact}"
                    }

                    if (isBleedingEdge()) {
                        def artifact = "h2o4gpu-${versionTag}-py36-none-any.whl"
                        def localArtifact = "src/interface_py/dist4/${artifact}"
                        def bucket = "s3://artifacts.h2o.ai/releases/bleeding-edge/ai/h2o/h2o4gpu/${versionTag}_nccl_cuda9/"
                        sh "s3cmd put ${localArtifact} ${bucket}"
                        sh "s3cmd setacl --acl-public  ${bucket}${artifact}"
                    }
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
                        rm -rf build/VERSION.txt ; make build/VERSION.txt
                    """
                    def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                    CONTAINER_NAME = "h2o4gpu-nccl-cuda9-runtime-${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        sh """
                                nvidia-docker build  -t opsh2oai/h2o4gpu-nccl-cuda9-runtime:latest -f Dockerfile-runtime --build-arg cuda=nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04 --build-arg wheel=${versionTag}_nccl_cuda9/h2o4gpu-${versionTag}-py36-none-any.whl .
                                nvidia-docker run  --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-nccl-cuda9-runtime
                                nvidia-docker exec ${CONTAINER_NAME} rm -rf data
                                nvidia-docker exec ${CONTAINER_NAME} ln -s /data ./data
                                nvidia-docker exec ${CONTAINER_NAME} rm -rf open_data
                                nvidia-docker exec ${CONTAINER_NAME} ln -s /open_data ./open_data
                                nvidia-docker exec ${CONTAINER_NAME} cd /jupyter/demos ; python -c "exec(\"from sklearn.datasets import fetch_covtype\\ncov = fetch_covtype()\")"
                                nvidia-docker exec ${CONTAINER_NAME} cd /jupyter/demos ; cp /open_data/creditcard.csv .
                                nvidia-docker exec ${CONTAINER_NAME} cd /jupyter/demos ; wget https://s3.amazonaws.com/h2o-public-test-data/h2o4gpu/open_data/kmeans_data/h2o-logo.jpg
                                nvidia-docker exec ${CONTAINER_NAME} cd /jupyter/demos ; cp /home/0xdiag/h2o4gpu/data/ipums_1k.csv .
                                nvidia-docker exec ${CONTAINER_NAME} cd /jupyter/demos ; cp /home/0xdiag/h2o4gpu/data/ipums.feather .
                                nvidia-docker save opsh2oai/h2o4gpu-nccl-cuda9-runtime > h2o4gpu-nccl-cuda9-runtime.tar
                                gzip  h2o4gpu-nccl-cuda9-runtime.tar
                            """
                        stash includes: 'h2o4gpu-nccl-cuda9-runtime.tar.gz', name: 'docker-nccl-cuda9-runtime'
                        // Archive artifacts
                        arch 'h2o4gpu-nccl-cuda9-runtime.tar.gz'
                    }
                }
            }
        }

        stage('Publish Runtime Docker for nccl CUDA9 to S3') {
            agent {
                label "linux"
            }

            steps {
                unstash 'docker-cuda9-runtime'
                unstash 'version_info'
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                    script {
                        // Load the version file content
                        def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                        def version = utilsLib.fragmentVersion(versionTag)
                        def _majorVersion = version[0]
                        def _buildVersion = version[1]
                        version = null // This is necessary, else version:Tuple will be serialized

                        if (isRelease()) {
                            s3up {
                                localArtifact = 'h2o4gpu-nccl-cuda9-runtime.tar.gz'
                                artifactId = "h2o4gpu"
                                majorVersion = _majorVersion
                                buildVersion = _buildVersion
                                keepPrivate = false
                                remoteArtifactBucket = "s3://artifacts.h2o.ai/releases/stable"
                            }
                            sh "s3cmd setacl --acl-public s3://artifacts.h2o.ai/releases/stable/ai/h2o/h2o4gpu/${versionTag}_nccl_cuda9/h2o4gpu-nccl-cuda9-runtime.tar.gz"
                        }

                        if (isBleedingEdge()) {
                            s3up {
                                localArtifact = 'h2o4gpu-nccl-cuda9-runtime.tar.gz'
                                artifactId = "h2o4gpu"
                                majorVersion = _majorVersion
                                buildVersion = _buildVersion
                                keepPrivate = false
                                remoteArtifactBucket = "s3://artifacts.h2o.ai/releases/bleeding-edge"
                            }
                            sh "s3cmd setacl --acl-public s3://artifacts.h2o.ai/releases/bleeding-edge/ai/h2o/h2o4gpu/${versionTag}_nccl_cuda9/h2o4gpu-nccl-cuda9-runtime.tar.gz"
                        }
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
                    CONTAINER_NAME = "h2o4gpu-nonccl-cuda9-${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        try {
                            sh """
                                    nvidia-docker build  -t opsh2oai/h2o4gpu-build -f Dockerfile-build --build-arg cuda=nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 .
                                    nvidia-docker run --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-build
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
                unstash 'linux_whl3'
                unstash 'version_info'
                sh 'echo "Stashed files:" && ls -l src/interface_py/dist3/'
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                script {
                    // Load the version file content
                    def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                    def version = utilsLib.fragmentVersion(versionTag)
                    def _majorVersion = version[0]
                    def _buildVersion = version[1]
                    version = null // This is necessary, else version:Tuple will be serialized

                    if (isRelease()) {
                        def artifact = "h2o4gpu-${versionTag}-py36-none-any.whl"
                        def localArtifact = "src/interface_py/dist3/${artifact}"
                        def bucket = "s3://artifacts.h2o.ai/releases/stable/ai/h2o/h2o4gpu/${versionTag}_nonccl_cuda9/"
                        sh "s3cmd put ${localArtifact} ${bucket}"
                        sh "s3cmd setacl --acl-public  ${bucket}${artifact}"
                    }

                    if (isBleedingEdge()) {
                        def artifact = "h2o4gpu-${versionTag}-py36-none-any.whl"
                        def localArtifact = "src/interface_py/dist3/${artifact}"
                        def bucket = "s3://artifacts.h2o.ai/releases/bleeding-edge/ai/h2o/h2o4gpu/${versionTag}_nonccl_cuda9/"
                        sh "s3cmd put ${localArtifact} ${bucket}"
                        sh "s3cmd setacl --acl-public  ${bucket}${artifact}"
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
                        rm -rf build/VERSION.txt ; make build/VERSION.txt
                    """
                    def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                    CONTAINER_NAME = "h2o4gpu-nonccl-cuda9-runtime-${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        sh """
                                nvidia-docker build  -t opsh2oai/h2o4gpu-nonccl-cuda9-runtime:latest -f Dockerfile-runtime --build-arg cuda=nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04  --build-arg wheel=${versionTag}_nonccl_cuda9/h2o4gpu-${versionTag}-py36-none-any.whl .
                                nvidia-docker run  --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-nonccl-cuda9-runtime
                                nvidia-docker exec ${CONTAINER_NAME} rm -rf data
                                nvidia-docker exec ${CONTAINER_NAME} ln -s /data ./data
                                nvidia-docker exec ${CONTAINER_NAME} rm -rf open_data
                                nvidia-docker exec ${CONTAINER_NAME} ln -s /open_data ./open_data
                                nvidia-docker exec ${CONTAINER_NAME} cd /jupyter/demos ; python -c "exec(\"from sklearn.datasets import fetch_covtype\\ncov = fetch_covtype()\")"
                                nvidia-docker exec ${CONTAINER_NAME} cd /jupyter/demos ; cp /open_data/creditcard.csv .
                                nvidia-docker exec ${CONTAINER_NAME} cd /jupyter/demos ; wget https://s3.amazonaws.com/h2o-public-test-data/h2o4gpu/open_data/kmeans_data/h2o-logo.jpg
                                nvidia-docker exec ${CONTAINER_NAME} cd /jupyter/demos ; cp /home/0xdiag/h2o4gpu/data/ipums_1k.csv .
                                nvidia-docker exec ${CONTAINER_NAME} cd /jupyter/demos ; cp /home/0xdiag/h2o4gpu/data/ipums.feather .
                                nvidia-docker save opsh2oai/h2o4gpu-nonccl-cuda9-runtime > h2o4gpu-nonccl-cuda9-runtime.tar
                                gzip  h2o4gpu-nonccl-cuda9-runtime.tar
                            """
                        stash includes: 'h2o4gpu-nonccl-cuda9-runtime.tar.gz', name: 'docker-nonccl-cuda9-runtime'
                        // Archive artifacts
                        arch 'h2o4gpu-nonccl-cuda9-runtime.tar.gz'
                    }
                }
            }
        }

        stage('Publish Runtime Docker for nonccl CUDA9 to S3') {
            agent {
                label "linux"
            }

            steps {
                unstash 'docker-cuda9-runtime'
                unstash 'version_info'
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                    script {
                        // Load the version file content
                        def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                        def version = utilsLib.fragmentVersion(versionTag)
                        def _majorVersion = version[0]
                        def _buildVersion = version[1]
                        version = null // This is necessary, else version:Tuple will be serialized

                        if (isRelease()) {
                            s3up {
                                localArtifact = 'h2o4gpu-nonccl-cuda9-runtime.tar.gz'
                                artifactId = "h2o4gpu"
                                majorVersion = _majorVersion
                                buildVersion = _buildVersion
                                keepPrivate = false
                                remoteArtifactBucket = "s3://artifacts.h2o.ai/releases/stable"
                            }
                            sh "s3cmd setacl --acl-public s3://artifacts.h2o.ai/releases/stable/ai/h2o/h2o4gpu/${versionTag}_nonccl_cuda9/h2o4gpu-nonccl-cuda9-runtime.tar.gz"
                        }

                        if (isBleedingEdge()) {
                            s3up {
                                localArtifact = 'h2o4gpu-nonccl-cuda9-runtime.tar.gz'
                                artifactId = "h2o4gpu"
                                majorVersion = _majorVersion
                                buildVersion = _buildVersion
                                keepPrivate = false
                                remoteArtifactBucket = "s3://artifacts.h2o.ai/releases/bleeding-edge"
                            }
                            sh "s3cmd setacl --acl-public s3://artifacts.h2o.ai/releases/bleeding-edge/ai/h2o/h2o4gpu/${versionTag}_nonccl_cuda9/h2o4gpu-nonccl-cuda9-runtime.tar.gz"
                        }
                    }
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

def isRelease() {
    return env.BRANCH_NAME.startsWith("rel")
}

def isBleedingEdge() {
    return env.BRANCH_NAME.startsWith("master")
}
