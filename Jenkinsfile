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
                    CONTAINER_NAME = 'h2o4gpu-${extratag}-${SAFE_CHANGE_ID}-${env.BUILD_ID}'
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
                script {
                    def extratag = "_nccl_cuda8"
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
        stage('Pylint on Linux nccl CUDA8') {
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
                script {
                    def extratag = "_nccl_cuda8"
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
        }


        stage('Publish to S3 nccl CUDA8') {
            agent {
                label "linux"
            }

            steps {
                unstash 'version_info'
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                    script {
                        def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                        def extratag = "_nccl_cuda8"
                        unstash 'linux_whl'
                        sh 'echo "Stashed files:" && ls -l src/interface_py/dist/'
                        def artifactId = h2o4gpu
                        def artifact = "${artifactId}-${versionTag}-py36-none-any.whl"
                        def localArtifact = "src/interface_py/dist/${artifact}"
                        s3up_simple(${versionTag}, ${extratag}, ${artifactId}, ${artifact}, ${localArtifact})
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

