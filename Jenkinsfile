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

// Just Notes:
//def jobnums       = [0 , 1 , 2  , 3]
//def tags          = ["nccl" , "nonccl" , "nccl"  , "nonccl"]
//def cudatags      = ["cuda8", "cuda8"  , "cuda9" , "cuda9"]
//def dobuilds      = [1, 0, 0, 0]
//def dofulltests   = [1, 0, 0, 0]
//def dopytests     = [1, 0, 0, 0]
//def doruntimes    = [1, 1, 1, 1]
//def dockerimagesbuild    = ["nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04", "nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04", "nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04", "nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04"]
//def dockerimagesruntime  = ["nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04", "nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04", "nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04", "nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04"]
//def dists         = ["dist","dist2","dist3","dist4"]


// MAJOR NOTE: all other nonccl-cuda8, nccl-cuda9, nonccl-cuda9 are just copies of nccl-cuda8 but with test as fast



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



        /////////////////////////////////////////////////////////////////////
        //
        // -nccl-cuda8
        //
        //  Avoid mr-dl8 and mr-dl10 for build for now due to permission denied issue
        /////////////////////////////////////////////////////////////////////
        stage("Build Wheel on Linux -nccl-cuda8") {

            agent {
                label "nvidia-docker && (mr-dl11 || mr-dl16)"
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
                    def tag = "nccl"
                    def cudatag = "cuda8"
                    def dist = "dist"
                    def dockerimage = "nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04"
                    // derived tag
                    def extratag = "-${tag}-${cudatag}"
                    CONTAINER_NAME = "h2o4gpu-build-${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        sh """
                                nvidia-docker build  -t opsh2oai/h2o4gpu-${extratag}-build -f Dockerfile-build --rm=false --build-arg cuda=${dockerimage} .
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
                            nvidia-docker stop ${CONTAINER_NAME}
                            """
                        stash includes: "src/interface_py/${dist}/*.whl", name: 'linux_whl1'
                        stash includes: 'build/VERSION.txt', name: 'version_info'
                        // Archive artifacts
                        arch "src/interface_py/${dist}/*.whl"
                    }
                }
            }
        }



        stage("Full Test Wheel & Pylint & S3up on Linux -nccl-cuda8") {
            agent {
                label "gpu && nvidia-docker && (mr-dl11 || mr-dl16)"
            }
            steps {
                dumpInfo 'Linux Test Info'
                // Get source code (should put tests into wheel, then wouldn't have to checkout)
                retryWithTimeout(100 /* seconds */, 3 /* retries */) {
                    checkout scm
                }
                script {
                    unstash 'version_info'
                    def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                    unstash 'linux_whl1'

                    def tag = "nccl"
                    def cudatag = "cuda8"
                    def dist = "dist"
                    def dockerimage = "nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04"
                    def extratag = "-${tag}-${cudatag}"
                    CONTAINER_NAME = "h2o4gpu-build-${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    try {
                        sh """
                            nvidia-docker build  -t opsh2oai/h2o4gpu-${extratag}-build -f Dockerfile-build --rm=false --build-arg cuda=${dockerimage} .
                            nvidia-docker run  --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-${extratag}-build
                            nvidia-docker exec ${CONTAINER_NAME} rm -rf data
                            nvidia-docker exec ${CONTAINER_NAME} ln -s /data ./data
                            nvidia-docker exec ${CONTAINER_NAME} rm -rf open_data
                            nvidia-docker exec ${CONTAINER_NAME} ln -s /open_data ./open_data
                            nvidia-docker exec ${CONTAINER_NAME} rm -rf py3nvml
                            nvidia-docker exec ${CONTAINER_NAME} bash -c 'export HOME=`pwd`; eval \"\$(/root/.pyenv/bin/pyenv init -)\"  ; /root/.pyenv/bin/pyenv global 3.6.1; pip install `find src/interface_py/${dist} -name "*h2o4gpu*.whl"`; make dotest'
                            nvidia-docker exec ${CONTAINER_NAME} touch src/interface_py/h2o4gpu/__init__.py
                            nvidia-docker exec ${CONTAINER_NAME} bash -c 'eval \"\$(/root/.pyenv/bin/pyenv init -)\"  ;  /root/.pyenv/bin/pyenv global 3.6.1; make pylint'
                            nvidia-docker stop ${CONTAINER_NAME}
                        """
                    } finally {
                        arch 'tmp/*.log'
                        junit testResults: 'build/test-reports/*.xml', keepLongStdio: true, allowEmptyResults: false
                    }
                }
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        script {
                            def tag = "nccl"
                            def cudatag = "cuda8"
                            def dist = "dist"
                            def extratag = "-${tag}-${cudatag}"
                            def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                            def artifactId = "h2o4gpu"
                            def artifact = "${artifactId}-${versionTag}-py36-none-any.whl"
                            def localArtifact = "src/interface_py/${dist}/${artifact}"
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
                            //if (!(isRelease() || isBleedingEdge())) {
                                // always upload for testing
                                def bucket = "s3://artifacts.h2o.ai/snapshots/ai/h2o/${artifactId}/${versionTag}${extratag}/"
                                sh "s3cmd put ${localArtifact} ${bucket}"
                                sh "s3cmd setacl --acl-public  ${bucket}${artifact}"
                            //}
                        }
                    }
                }
            }
        }


        stage("Build/Publish Runtime Docker -nccl-cuda8") {
            agent {
                label "nvidia-docker"
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
                unstash 'linux_whl1'
                unstash 'version_info'
                script {
                    sh 'echo "Stashed version file:" && ls -l build/'
                }
                script {
                    def tag = "nccl"
                    def cudatag = "cuda8"
                    def dockerimage = "nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04"
                    // derived tag
                    def extratag = "-${tag}-${cudatag}"
                    def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                    CONTAINER_NAME = "h2o4gpu-runtime-${SAFE_CHANGE_ID}-${env.BUILD_ID}"

                    //if (isRelease()) {
                    //    def buckettype = "releases/stable"
                    //} else if (isBleedingEdge()) {
                    //    def buckettype = "releases/bleeding-edge"
                    //} else {
                    //    def buckettype = "snapshots"
                    //}
                    def buckettype = "snapshots"

                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        sh """
                                nvidia-docker build -t opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime:latest -f Dockerfile-runtime --rm=false --build-arg cuda=${dockerimage} --build-arg wheel=${versionTag}${extratag}/h2o4gpu-${versionTag}-py36-none-any.whl --build-arg buckettype=${buckettype} .
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
                                nvidia-docker stop ${CONTAINER_NAME}
                                nvidia-docker save opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime | gzip > h2o4gpu-${versionTag}${extratag}-runtime.tar.gz
                            """
                    }
                }
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                    script {
                        def tag = "nccl"
                        def cudatag = "cuda8"
                        def extratag = "-${tag}-${cudatag}"
                        def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                        def artifactId = "h2o4gpu"
                        def artifact = "${artifactId}-${versionTag}${extratag}-runtime.tar.gz"
                        def localArtifact = "${artifact}"
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
                            def bucket = "s3://artifacts.h2o.ai/snapshots/bleeding-edge/ai/h2o/${artifactId}/${versionTag}${extratag}/"
                            sh "s3cmd put ${localArtifact} ${bucket}"
                            sh "s3cmd setacl --acl-public  ${bucket}${artifact}"
                        }
                    }
                }
                }
            }
        }








        /////////////////////////////////////////////////////////////////////
        //
        // -nonccl-cuda8
        //
        //  Avoid mr-dl8 and mr-dl10 for build for now due to permission denied issue
        /////////////////////////////////////////////////////////////////////
        stage("Build Wheel on Linux -nonccl-cuda8") {

            agent {
                label "nvidia-docker && (mr-dl11 || mr-dl16)"
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
                    def tag = "nonccl"
                    def cudatag = "cuda8"
                    def dist = "dist"
                    def dockerimage = "nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04"
                    // derived tag
                    def extratag = "-${tag}-${cudatag}"
                    CONTAINER_NAME = "h2o4gpu-build-${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        sh """
                                nvidia-docker build  -t opsh2oai/h2o4gpu-${extratag}-build -f Dockerfile-build --rm=false --build-arg cuda=${dockerimage} .
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
                            nvidia-docker stop ${CONTAINER_NAME}
                            """
                        stash includes: "src/interface_py/${dist}/*.whl", name: 'linux_whl2'
                        stash includes: 'build/VERSION.txt', name: 'version_info'
                        // Archive artifacts
                        arch "src/interface_py/${dist}/*.whl"
                    }
                }
            }
        }



        stage("Fast Test Wheel & Pylint & S3up on Linux -nonccl-cuda8") {
            agent {
                label "gpu && nvidia-docker && (mr-dl11 || mr-dl16)"
            }
            steps {
                dumpInfo 'Linux Test Info'
                // Get source code (should put tests into wheel, then wouldn't have to checkout)
                retryWithTimeout(100 /* seconds */, 3 /* retries */) {
                    checkout scm
                }
                script {
                    unstash 'version_info'
                    def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                    unstash 'linux_whl2'
                    def tag = "nonccl"
                    def cudatag = "cuda8"
                    def dist = "dist"
                    def dockerimage = "nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04"
                    def extratag = "-${tag}-${cudatag}"
                    CONTAINER_NAME = "h2o4gpu-build-${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    try {
                        sh """
                            nvidia-docker build  -t opsh2oai/h2o4gpu-${extratag}-build -f Dockerfile-build --rm=false --build-arg cuda=${dockerimage} .
                            nvidia-docker run  --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-${extratag}-build
                            nvidia-docker exec ${CONTAINER_NAME} rm -rf data
                            nvidia-docker exec ${CONTAINER_NAME} ln -s /data ./data
                            nvidia-docker exec ${CONTAINER_NAME} rm -rf open_data
                            nvidia-docker exec ${CONTAINER_NAME} ln -s /open_data ./open_data
                            nvidia-docker exec ${CONTAINER_NAME} rm -rf py3nvml
                            nvidia-docker exec ${CONTAINER_NAME} bash -c 'export HOME=`pwd`; eval \"\$(/root/.pyenv/bin/pyenv init -)\"  ; /root/.pyenv/bin/pyenv global 3.6.1; pip install `find src/interface_py/${dist} -name "*h2o4gpu*.whl"`; make dotestfast'
                            nvidia-docker exec ${CONTAINER_NAME} touch src/interface_py/h2o4gpu/__init__.py
                            nvidia-docker exec ${CONTAINER_NAME} bash -c 'eval \"\$(/root/.pyenv/bin/pyenv init -)\"  ;  /root/.pyenv/bin/pyenv global 3.6.1; make pylint'
                            nvidia-docker stop ${CONTAINER_NAME}
                        """
                    } finally {
                        arch 'tmp/*.log'
                        junit testResults: 'build/test-reports/*.xml', keepLongStdio: true, allowEmptyResults: false
                    }
                }
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        script {
                            def tag = "nonccl"
                            def cudatag = "cuda8"
                            def dist = "dist"
                            def extratag = "-${tag}-${cudatag}"
                            def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                            def artifactId = "h2o4gpu"
                            def artifact = "${artifactId}-${versionTag}-py36-none-any.whl"
                            def localArtifact = "src/interface_py/${dist}/${artifact}"
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
                            //if (!(isRelease() || isBleedingEdge())) {
                                // always upload for testing
                                def bucket = "s3://artifacts.h2o.ai/snapshots/ai/h2o/${artifactId}/${versionTag}${extratag}/"
                                sh "s3cmd put ${localArtifact} ${bucket}"
                                sh "s3cmd setacl --acl-public  ${bucket}${artifact}"
                            //}
                        }
                    }
                }
            }
        }


        stage("Build/Publish Runtime Docker -nonccl-cuda8") {
            agent {
                label "nvidia-docker"
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
                unstash 'linux_whl2'
                unstash 'version_info'
                script {
                    sh 'echo "Stashed version file:" && ls -l build/'
                }
                script {
                    def tag = "nonccl"
                    def cudatag = "cuda8"
                    def dockerimage = "nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04"
                    // derived tag
                    def extratag = "-${tag}-${cudatag}"
                    def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                    CONTAINER_NAME = "h2o4gpu-runtime-${SAFE_CHANGE_ID}-${env.BUILD_ID}"

                    //if (isRelease()) {
                    //    def buckettype = "releases/stable"
                    //} else if (isBleedingEdge()) {
                    //    def buckettype = "releases/bleeding-edge"
                    //} else {
                    //    def buckettype = "snapshots"
                    //}
                    def buckettype = "snapshots"

                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        sh """
                                nvidia-docker build -t opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime:latest -f Dockerfile-runtime --rm=false --build-arg cuda=${dockerimage} --build-arg wheel=${versionTag}${extratag}/h2o4gpu-${versionTag}-py36-none-any.whl --build-arg buckettype=${buckettype} .
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
                                nvidia-docker stop ${CONTAINER_NAME}
                                nvidia-docker save opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime | gzip > h2o4gpu-${versionTag}${extratag}-runtime.tar.gz
                            """
                    }
                }
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                    script {
                        def tag = "nonccl"
                        def cudatag = "cuda8"
                        def extratag = "-${tag}-${cudatag}"
                        def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                        def artifactId = "h2o4gpu"
                        def artifact = "${artifactId}-${versionTag}${extratag}-runtime.tar.gz"
                        def localArtifact = "${artifact}"
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
                            def bucket = "s3://artifacts.h2o.ai/snapshots/bleeding-edge/ai/h2o/${artifactId}/${versionTag}${extratag}/"
                            sh "s3cmd put ${localArtifact} ${bucket}"
                            sh "s3cmd setacl --acl-public  ${bucket}${artifact}"
                        }
                    }
                }
                }
            }
        }










        /////////////////////////////////////////////////////////////////////
        //
        // -nccl-cuda9
        //
        //  Avoid mr-dl8 and mr-dl10 for build for now due to permission denied issue
        /////////////////////////////////////////////////////////////////////
        stage("Build Wheel on Linux -nccl-cuda9") {

            agent {
                label "nvidia-docker && (mr-dl11 || mr-dl16)"
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
                    def tag = "nccl"
                    def cudatag = "cuda9"
                    def dist = "dist"
                    def dockerimage = "nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04"
                    // derived tag
                    def extratag = "-${tag}-${cudatag}"
                    CONTAINER_NAME = "h2o4gpu-build-${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        sh """
                                nvidia-docker build  -t opsh2oai/h2o4gpu-${extratag}-build -f Dockerfile-build --rm=false --build-arg cuda=${dockerimage} .
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
                            nvidia-docker stop ${CONTAINER_NAME}
                            """
                        stash includes: "src/interface_py/${dist}/*.whl", name: 'linux_whl3'
                        stash includes: 'build/VERSION.txt', name: 'version_info'
                        // Archive artifacts
                        arch "src/interface_py/${dist}/*.whl"
                    }
                }
            }
        }



        stage("Fast Test Wheel & Pylint & S3up on Linux -nccl-cuda9") {
            agent {
                label "gpu && nvidia-docker && (mr-dl11 || mr-dl16)"
            }
            steps {
                dumpInfo 'Linux Test Info'
                // Get source code (should put tests into wheel, then wouldn't have to checkout)
                retryWithTimeout(100 /* seconds */, 3 /* retries */) {
                    checkout scm
                }
                script {
                    unstash 'version_info'
                    def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                    unstash 'linux_whl3'

                    def tag = "nccl"
                    def cudatag = "cuda9"
                    def dist = "dist"
                    def dockerimage = "nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04"
                    def extratag = "-${tag}-${cudatag}"
                    CONTAINER_NAME = "h2o4gpu-build-${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    try {
                        sh """
                            nvidia-docker build  -t opsh2oai/h2o4gpu-${extratag}-build -f Dockerfile-build --rm=false --build-arg cuda=${dockerimage} .
                            nvidia-docker run  --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-${extratag}-build
                            nvidia-docker exec ${CONTAINER_NAME} rm -rf data
                            nvidia-docker exec ${CONTAINER_NAME} ln -s /data ./data
                            nvidia-docker exec ${CONTAINER_NAME} rm -rf open_data
                            nvidia-docker exec ${CONTAINER_NAME} ln -s /open_data ./open_data
                            nvidia-docker exec ${CONTAINER_NAME} rm -rf py3nvml
                            nvidia-docker exec ${CONTAINER_NAME} bash -c 'export HOME=`pwd`; eval \"\$(/root/.pyenv/bin/pyenv init -)\"  ; /root/.pyenv/bin/pyenv global 3.6.1; pip install `find src/interface_py/${dist} -name "*h2o4gpu*.whl"`; make dotestfast'
                            nvidia-docker exec ${CONTAINER_NAME} touch src/interface_py/h2o4gpu/__init__.py
                            nvidia-docker exec ${CONTAINER_NAME} bash -c 'eval \"\$(/root/.pyenv/bin/pyenv init -)\"  ;  /root/.pyenv/bin/pyenv global 3.6.1; make pylint'
                            nvidia-docker stop ${CONTAINER_NAME}
                        """
                    } finally {
                        arch 'tmp/*.log'
                        junit testResults: 'build/test-reports/*.xml', keepLongStdio: true, allowEmptyResults: false
                    }
                }
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        script {
                            def tag = "nccl"
                            def cudatag = "cuda9"
                            def dist = "dist"
                            def extratag = "-${tag}-${cudatag}"
                            def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                            def artifactId = "h2o4gpu"
                            def artifact = "${artifactId}-${versionTag}-py36-none-any.whl"
                            def localArtifact = "src/interface_py/${dist}/${artifact}"
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
                            //if (!(isRelease() || isBleedingEdge())) {
                                // always upload for testing
                                def bucket = "s3://artifacts.h2o.ai/snapshots/ai/h2o/${artifactId}/${versionTag}${extratag}/"
                                sh "s3cmd put ${localArtifact} ${bucket}"
                                sh "s3cmd setacl --acl-public  ${bucket}${artifact}"
                            //}
                        }
                    }
                }
            }
        }


        stage("Build/Publish Runtime Docker -nccl-cuda9") {
            agent {
                label "nvidia-docker"
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
                unstash 'linux_whl3'
                unstash 'version_info'
                script {
                    sh 'echo "Stashed version file:" && ls -l build/'
                }
                script {
                    def tag = "nccl"
                    def cudatag = "cuda9"
                    def dockerimage = "nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04"
                    // derived tag
                    def extratag = "-${tag}-${cudatag}"
                    def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                    CONTAINER_NAME = "h2o4gpu-runtime-${SAFE_CHANGE_ID}-${env.BUILD_ID}"

                    //if (isRelease()) {
                    //    def buckettype = "releases/stable"
                    //} else if (isBleedingEdge()) {
                    //    def buckettype = "releases/bleeding-edge"
                    //} else {
                    //    def buckettype = "snapshots"
                    //}
                    def buckettype = "snapshots"

                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        sh """
                                nvidia-docker build -t opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime:latest -f Dockerfile-runtime --rm=false --build-arg cuda=${dockerimage} --build-arg wheel=${versionTag}${extratag}/h2o4gpu-${versionTag}-py36-none-any.whl --build-arg buckettype=${buckettype} .
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
                                nvidia-docker stop ${CONTAINER_NAME}
                                nvidia-docker save opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime | gzip > h2o4gpu-${versionTag}${extratag}-runtime.tar.gz
                            """
                    }
                }
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                    script {
                        def tag = "nccl"
                        def cudatag = "cuda9"
                        def extratag = "-${tag}-${cudatag}"
                        def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                        def artifactId = "h2o4gpu"
                        def artifact = "${artifactId}-${versionTag}${extratag}-runtime.tar.gz"
                        def localArtifact = "${artifact}"
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
                            def bucket = "s3://artifacts.h2o.ai/snapshots/bleeding-edge/ai/h2o/${artifactId}/${versionTag}${extratag}/"
                            sh "s3cmd put ${localArtifact} ${bucket}"
                            sh "s3cmd setacl --acl-public  ${bucket}${artifact}"
                        }
                    }
                }
                }
            }
        }








        /////////////////////////////////////////////////////////////////////
        //
        // -nonccl-cuda9
        //
        //  Avoid mr-dl8 and mr-dl10 for build for now due to permission denied issue
        /////////////////////////////////////////////////////////////////////
        stage("Build Wheel on Linux -nonccl-cuda9") {

            agent {
                label "nvidia-docker && (mr-dl11 || mr-dl16)"
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
                    def tag = "nonccl"
                    def cudatag = "cuda9"
                    def dist = "dist"
                    def dockerimage = "nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04"
                    // derived tag
                    def extratag = "-${tag}-${cudatag}"
                    CONTAINER_NAME = "h2o4gpu-build-${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        sh """
                                nvidia-docker build  -t opsh2oai/h2o4gpu-${extratag}-build -f Dockerfile-build --rm=false --build-arg cuda=${dockerimage} .
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
                            nvidia-docker stop ${CONTAINER_NAME}
                            """
                        stash includes: "src/interface_py/${dist}/*.whl", name: 'linux_whl4'
                        stash includes: 'build/VERSION.txt', name: 'version_info'
                        // Archive artifacts
                        arch "src/interface_py/${dist}/*.whl"
                    }
                }
            }
        }



        stage("Fast Test Wheel & Pylint & S3up on Linux -nonccl-cuda9") {
            agent {
                label "gpu && nvidia-docker && (mr-dl11 || mr-dl16)"
            }
            steps {
                dumpInfo 'Linux Test Info'
                // Get source code (should put tests into wheel, then wouldn't have to checkout)
                retryWithTimeout(100 /* seconds */, 3 /* retries */) {
                    checkout scm
                }
                script {
                    unstash 'version_info'
                    def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                    unstash 'linux_whl4'

                    def tag = "nonccl"
                    def cudatag = "cuda9"
                    def dist = "dist"
                    def dockerimage = "nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04"
                    def extratag = "-${tag}-${cudatag}"
                    CONTAINER_NAME = "h2o4gpu-build-${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    try {
                        sh """
                            nvidia-docker build  -t opsh2oai/h2o4gpu-${extratag}-build -f Dockerfile-build --rm=false --build-arg cuda=${dockerimage} .
                            nvidia-docker run  --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-${extratag}-build
                            nvidia-docker exec ${CONTAINER_NAME} rm -rf data
                            nvidia-docker exec ${CONTAINER_NAME} ln -s /data ./data
                            nvidia-docker exec ${CONTAINER_NAME} rm -rf open_data
                            nvidia-docker exec ${CONTAINER_NAME} ln -s /open_data ./open_data
                            nvidia-docker exec ${CONTAINER_NAME} rm -rf py3nvml
                            nvidia-docker exec ${CONTAINER_NAME} bash -c 'export HOME=`pwd`; eval \"\$(/root/.pyenv/bin/pyenv init -)\"  ; /root/.pyenv/bin/pyenv global 3.6.1; pip install `find src/interface_py/${dist} -name "*h2o4gpu*.whl"`; make dotestfast'
                            nvidia-docker exec ${CONTAINER_NAME} touch src/interface_py/h2o4gpu/__init__.py
                            nvidia-docker exec ${CONTAINER_NAME} bash -c 'eval \"\$(/root/.pyenv/bin/pyenv init -)\"  ;  /root/.pyenv/bin/pyenv global 3.6.1; make pylint'
                            nvidia-docker stop ${CONTAINER_NAME}
                        """
                    } finally {
                        arch 'tmp/*.log'
                        junit testResults: 'build/test-reports/*.xml', keepLongStdio: true, allowEmptyResults: false
                    }
                }
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        script {
                            def tag = "nonccl"
                            def cudatag = "cuda9"
                            def dist = "dist"
                            def extratag = "-${tag}-${cudatag}"
                            def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                            def artifactId = "h2o4gpu"
                            def artifact = "${artifactId}-${versionTag}-py36-none-any.whl"
                            def localArtifact = "src/interface_py/${dist}/${artifact}"
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
                            //if (!(isRelease() || isBleedingEdge())) {
                                // always upload for testing
                                def bucket = "s3://artifacts.h2o.ai/snapshots/ai/h2o/${artifactId}/${versionTag}${extratag}/"
                                sh "s3cmd put ${localArtifact} ${bucket}"
                                sh "s3cmd setacl --acl-public  ${bucket}${artifact}"
                            //}
                        }
                    }
                }
            }
        }


        stage("Build/Publish Runtime Docker -nonccl-cuda9") {
            agent {
                label "nvidia-docker"
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
                unstash 'linux_whl4'
                unstash 'version_info'
                script {
                    sh 'echo "Stashed version file:" && ls -l build/'
                }
                script {
                    def tag = "nonccl"
                    def cudatag = "cuda9"
                    def dockerimage = "nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04"
                    // derived tag
                    def extratag = "-${tag}-${cudatag}"
                    def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                    CONTAINER_NAME = "h2o4gpu-runtime-${SAFE_CHANGE_ID}-${env.BUILD_ID}"

                    //if (isRelease()) {
                    //    def buckettype = "releases/stable"
                    //} else if (isBleedingEdge()) {
                    //    def buckettype = "releases/bleeding-edge"
                    //} else {
                    //    def buckettype = "snapshots"
                    //}
                    def buckettype = "snapshots"

                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        sh """
                                nvidia-docker build -t opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime:latest -f Dockerfile-runtime --rm=false --build-arg cuda=${dockerimage} --build-arg wheel=${versionTag}${extratag}/h2o4gpu-${versionTag}-py36-none-any.whl --build-arg buckettype=${buckettype} .
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
                                nvidia-docker stop ${CONTAINER_NAME}
                                nvidia-docker save opsh2oai/h2o4gpu-${versionTag}${extratag}-runtime | gzip > h2o4gpu-${versionTag}${extratag}-runtime.tar.gz
                            """
                    }
                }
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                    script {
                        def tag = "nonccl"
                        def cudatag = "cuda9"
                        def extratag = "-${tag}-${cudatag}"
                        def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                        def artifactId = "h2o4gpu"
                        def artifact = "${artifactId}-${versionTag}${extratag}-runtime.tar.gz"
                        def localArtifact = "${artifact}"
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
                            def bucket = "s3://artifacts.h2o.ai/snapshots/bleeding-edge/ai/h2o/${artifactId}/${versionTag}${extratag}/"
                            sh "s3cmd put ${localArtifact} ${bucket}"
                            sh "s3cmd setacl --acl-public  ${bucket}${artifact}"
                        }
                    }
                }
                }
            }
        }
















    } // end over stages
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

