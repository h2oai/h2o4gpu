#!/usr/bin/groovy
// TOOD: rename to @Library('h2o-jenkins-pipeline-lib') _
@Library('test-shared-library') _

import ai.h2o.ci.Utils

def utilsLib = new Utils()

SAFE_CHANGE_ID = changeId()
CONTAINER_NAME = "h2o4gpu-build-${SAFE_CHANGE_ID}-${env.BUILD_ID}"

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
                    buildOnLinux("nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04", "-nccl-cuda8", "dist")

                    buildInfo("h2o4gpu", isRelease())

                    script {
                        // Load the version file content
                        buildInfo.get().setVersion(utilsLib.getCommandOutput("cat build/VERSION.txt"))
                        utilsLib.setCurrentBuildName(buildInfo.get().getVersion())
                        utilsLib.appendBuildDescription("""|Authors: ${buildInfo.get().getAuthorNames().join(" ")}
                                |Git SHA: ${buildInfo.get().getGitSha().substring(0, 8)}
                                |""".stripMargin("|"))
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
                    unstash 'linux_whl1'
                    runTests("nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04", "-nccl-cuda8", "dist", "dotest")
                }
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        script {
                            publishToS3("-nccl-cuda8" , "dist")
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

                unstash 'linux_whl1'
                unstash 'version_info'
                script {
                    sh 'echo "Stashed version file:" && ls -l build/'
                }
                script {
                    buildRuntime("nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04", "-nccl-cuda8")
                }
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                    script {
                        publishRuntimeToS3("-nccl-cuda8")
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
                    buildOnLinux("nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04", "-nonccl-cuda8", "dist2")
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
                    unstash 'linux_whl2'
                    runTests("nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04", "-nonccl-cuda8", "dist2", "dotestfast")
                }
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        script {
                            publishToS3("-nonccl-cuda8" , "dist2")
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
                    buildRuntime("nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04", "-nonccl-cuda8")
                }
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                    script {
                        publishRuntimeToS3("-nonccl-cuda8")
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
                    buildOnLinux("nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04", "-nccl-cuda9", "dist4")
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
                    unstash 'linux_whl3'
                    runTests("nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04", "-nccl-cuda9", "dist4", "dotestfast")
                }
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        script {
                            publishToS3("-nccl-cuda9" , "dist4")
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
                    buildRuntime("nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04", "-nccl-cuda9")
                }
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                    script {
                        publishRuntimeToS3("-nccl-cuda9")
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
                    buildOnLinux("nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04", "-nonccl-cuda9", "dist3")
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
                    unstash 'linux_whl4'
                    runTests("nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04", "-nonccl-cuda9", "dist3", "dotestfast")
                }
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        script {
                            publishToS3("-nonccl-cuda9" , "dist3")
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
                    buildRuntime("nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04", "-nonccl-cuda9")
                }
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                    script {
                        publishRuntimeToS3("-nonccl-cuda9")
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
                                    [$class: 'DevelopersRecipientProvider'],
                            ]
                    )
                }
            }
        }
    }
}

@NonCPS
void publishToS3(String extratag, String dist) {
    def versionTag = buildInfo.get().getVersion()
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
    //}
}

@NonCPS
void publishRuntimeToS3(String extratag) {
    def versionTag = buildInfo.get().getVersion()
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
    //if (!(isRelease() || isBleedingEdge())) {
    // always upload for testing
    def bucket = "s3://artifacts.h2o.ai/snapshots/bleeding-edge/ai/h2o/${artifactId}/${versionTag}${extratag}/"
    sh "s3cmd put ${localArtifact} ${bucket}"
    //}
}

@NonCPS
void runTests(String dockerimage, String extratag, String dist, String target) {
    def versionTag = buildInfo.get().getVersion()

    try {
        sh """
            nvidia-docker build  -t opsh2oai/h2o4gpu-${extratag}-build -f Dockerfile-build --rm=false --build-arg cuda=${dockerimage} .
            nvidia-docker run  --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-${extratag}-build
            nvidia-docker exec ${CONTAINER_NAME} rm -rf data
            nvidia-docker exec ${CONTAINER_NAME} ln -s /data ./data
            nvidia-docker exec ${CONTAINER_NAME} rm -rf open_data
            nvidia-docker exec ${CONTAINER_NAME} ln -s /open_data ./open_data
            nvidia-docker exec ${CONTAINER_NAME} rm -rf py3nvml
            nvidia-docker exec ${CONTAINER_NAME} bash -c 'export HOME=`pwd`; eval \"\$(/root/.pyenv/bin/pyenv init -)\"  ; /root/.pyenv/bin/pyenv global 3.6.1; pip install `find src/interface_py/${dist} -name "*h2o4gpu-${versionTag}*.whl"`; make ${target}'
            nvidia-docker exec ${CONTAINER_NAME} touch src/interface_py/h2o4gpu/__init__.py
            nvidia-docker exec ${CONTAINER_NAME} bash -c 'eval \"\$(/root/.pyenv/bin/pyenv init -)\"  ;  /root/.pyenv/bin/pyenv global 3.6.1; make pylint'
            nvidia-docker stop ${CONTAINER_NAME}
            """
    } finally {
        arch 'tmp/*.log'
        junit testResults: 'build/test-reports/*.xml', keepLongStdio: true, allowEmptyResults: false
    }
}

@NonCPS
void buildOnLinux(String dockerimage, String extratag, String dist) {
    echo "Building on linux ${dockerimage} | ${extratag} | ${dist}"
    // Get source code
    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
        echo "Building on linux - running docker"
        sh """
            nvidia-docker build  -t opsh2oai/h2o4gpu-${extratag}-build -f Dockerfile-build --rm=false --build-arg cuda=${dockerimage} .
            nvidia-docker run --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-${extratag}-build
            nvidia-docker exec ${CONTAINER_NAME} rm -rf data
            nvidia-docker exec ${CONTAINER_NAME} ln -s /data ./data
            nvidia-docker exec ${CONTAINER_NAME} rm -rf open_data
            nvidia-docker exec ${CONTAINER_NAME} ln -s /open_data ./open_data
            nvidia-docker exec ${CONTAINER_NAME} bash -c 'eval \"\$(/root/.pyenv/bin/pyenv init -)\" ; /root/.pyenv/bin/pyenv global 3.6.1; ./scripts/gitshallow_submodules.sh; make ${env.MAKE_OPTS} fullinstalljenkins${extratag} H2O4GPU_BUILD=${env.BUILD_ID} H2O4GPU_SUFFIX=${isRelease() ? "" : "+" + utilsLib.getCiVersionSuffix()};'
            nvidia-docker stop ${CONTAINER_NAME}
           """

        stash includes: "src/interface_py/${dist}/*.whl", name: 'linux_whl1'
        stash includes: 'build/VERSION.txt', name: 'version_info'
        // Archive artifacts
        arch "src/interface_py/${dist}/*.whl"
    }
}

@NonCPS
void buildRuntime(String dockerimage, String extratag) {
    //if (isRelease()) {
    //    def buckettype = "releases/stable"
    //} else if (isBleedingEdge()) {
    //    def buckettype = "releases/bleeding-edge"
    //} else {
    //    def buckettype = "snapshots"
    //}
    def buckettype = "snapshots"
    def versionTag = buildInfo.get().getVersion()

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

def isRelease() {
    return env.BRANCH_NAME.startsWith("rel")
}

def isBleedingEdge() {
    return env.BRANCH_NAME.startsWith("master")
}

