#!/usr/bin/groovy
// TOOD: rename to @Library('h2o-jenkins-pipeline-lib') _
@Library('test-shared-library') _

import ai.h2o.ci.Utils

def utilsLib = new Utils()

String SAFE_CHANGE_ID = changeId()

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
        timeout(time: 60, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '10'))
        disableConcurrentBuilds()
    }

    environment {
        MAKE_OPTS = "-s CI=1" // -s: silent mode
    }

    stages {

        stage('Build on Linux') {
            agent {
                label "gpu && nvidia-docker"
            }

            steps {
                dumpInfo 'Linux Build Info'
                checkout([
                        $class                           : 'GitSCM',
                        branches                         : scm.branches,
                        doGenerateSubmoduleConfigurations: false,
                        extensions                       : scm.extensions + [[$class: 'SubmoduleOption', disableSubmodules: false, recursiveSubmodules: true, reference: '', trackingSubmodules: false]],
                        submoduleCfg                     : [],
                        userRemoteConfigs                : scm.userRemoteConfigs])
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                    sh """
                            nvidia-docker build -t opsh2oai/h2o4gpu-build -f Dockerfile-build .
                            nvidia-docker run --rm --name h2o4gpu${SAFE_CHANGE_ID}-$BUILD_ID -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-build
                            nvidia-docker exec h2o4gpu${SAFE_CHANGE_ID}-$BUILD_ID rm -rf data
                            nvidia-docker exec h2o4gpu${SAFE_CHANGE_ID}-$BUILD_ID ln -s /data ./data
                            nvidia-docker exec h2o4gpu${SAFE_CHANGE_ID}-$BUILD_ID rm -rf open_data
                            nvidia-docker exec h2o4gpu${SAFE_CHANGE_ID}-$BUILD_ID ln -s /open_data ./open_data
                            nvidia-docker exec h2o4gpu${SAFE_CHANGE_ID}-$BUILD_ID bash -c '. /h2oai_env/bin/activate; make ${env.MAKE_OPTS} AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} fullinstalljenkins; make build/VERSION.txt'
                            nvidia-docker stop h2o4gpu${SAFE_CHANGE_ID}-$BUILD_ID
                        """
                    stash includes: 'src/interface_py/dist/*.whl', name: 'linux_whl'
                    stash includes: 'build/VERSION.txt', name: 'version_info'
                    // Archive artifacts
                    arch 'src/interface_py/dist/*.whl'
                }
            }
        }

        stage('Test on Linux') {
            agent {
                label "mr-dl11"
            }

            steps {
                unstash 'linux_whl'
                dumpInfo 'Linux Test Info'
                script {
                    try {
                        sh """
                            nvidia-docker run --rm --name h2o4gpu${SAFE_CHANGE_ID}-$BUILD_ID -d -t -u `id -u`:`id -g` -v /home/0xdiag/h2o4gpu/data:/data -v /home/0xdiag/h2o4gpu/open_data:/open_data -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/h2o4gpu-build
                            nvidia-docker exec h2o4gpu${SAFE_CHANGE_ID}-$BUILD_ID rm -rf data
                            nvidia-docker exec h2o4gpu${SAFE_CHANGE_ID}-$BUILD_ID ln -s /data ./data
                            nvidia-docker exec h2o4gpu${SAFE_CHANGE_ID}-$BUILD_ID rm -rf open_data
                            nvidia-docker exec h2o4gpu${SAFE_CHANGE_ID}-$BUILD_ID ln -s /open_data ./open_data
                            nvidia-docker exec h2o4gpu${SAFE_CHANGE_ID}-$BUILD_ID rm -rf py3nvml
                            nvidia-docker exec h2o4gpu${SAFE_CHANGE_ID}-$BUILD_ID bash -c 'export HOME=`pwd`; . /h2oai_env/bin/activate; pip install `find src/interface_py/dist -name "*h2o4gpu*.whl"`; make dotest'
                        """
                    } finally {
                        sh """
                            nvidia-docker stop h2o4gpu${SAFE_CHANGE_ID}-$BUILD_ID
                        """
                        arch 'tmp/*.log'
                        junit testResults: 'build/test-reports/*.xml', keepLongStdio: true, allowEmptyResults: false
                        deleteDir()
                    }
                }
            }
        }

        stage('Publish to S3') {
            agent {
                label "linux"
            }

            steps {
                unstash 'linux_whl'
                unstash 'version_info'
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
                            localArtifact = 'src/interface_py/dist/*h2o4gpu*.whl'
                            artifactId = "h2o4gpu"
                            majorVersion = _majorVersion
                            buildVersion = _buildVersion
                            keepPrivate = false
                            remoteArtifactBucket = "s3://artifacts.h2o.ai/releases/stable"
                        }
                        sh "s3cmd setacl --acl-public s3://artifacts.h2o.ai/releases/stable/ai/h2o/h2o4gpu/${versionTag}/h2o4gpu-${versionTag}-py2.py3-none-any.whl"
                    }

                    if (isBleedingEdge()) {
                        s3up {
                            localArtifact = 'src/interface_py/dist/*h2o4gpu*.whl'
                            artifactId = "h2o4gpu"
                            majorVersion = _majorVersion
                            buildVersion = _buildVersion
                            keepPrivate = false
                            remoteArtifactBucket = "s3://artifacts.h2o.ai/releases/bleeding-edge"
                        }
                        sh "s3cmd setacl --acl-public s3://artifacts.h2o.ai/releases/bleeding-edge/ai/h2o/h2o4gpu/${versionTag}/h2o4gpu-${versionTag}-py2.py3-none-any.whl"
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