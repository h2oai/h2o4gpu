#!/usr/bin/groovy
// TOOD: rename to @Library('h2o-jenkins-pipeline-lib') _
@Library('test-shared-library') _

import ai.h2o.ci.Utils

def utilsLib = new Utils()

pipeline {
    agent none

    // Setup job options
    options {
        ansiColor('xterm')
        timestamps()
        timeout(time: 60, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }

    environment {
        MAKE_OPTS = "-s CI=1" // -s: silent mode
    }

    stages {

        stage('Build on Linux') {
            agent {
                dockerfile {
                    label "mr-dl11"
                    filename "Dockerfile-build"
                    args "-v /home/0xdiag/h2ogpuml/data:/data"
                }
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
                    rm -rf data
                    ln -s /data ./data
                    . /h2oai_env/bin/activate
                    make ${env.MAKE_OPTS} AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} fullinstalljenkins
                        """
                    stash includes: 'src/interface_py/dist/*.whl', name: 'linux_whl'
                    // Archive artifacts
                    arch 'src/interface_py/dist/*.whl'
                }
            }
        }
        stage('Test on Linux') {
            agent {
                dockerfile {
                    label "mr-dl11"
                    filename "Dockerfile-build"
                    args "-v /home/0xdiag/h2ogpuml/data:/data"
                }
            }
            steps {
                unstash 'linux_whl'
                dumpInfo 'Linux Test Info'
                script {
                    try {
                        sh """
                            rm -rf data
                            ln -s /data ./data
                            . /h2oai_env/bin/activate
                            pip install `find src/interface_py/dist -name "*h2ogpuml*.whl"`
                            make dotest
                        """
                    } finally {
                        arch 'tmp/*.log'
                        junit testResults: 'build/test-reports/*.xml', keepLongStdio: true, allowEmptyResults: false
                        deleteDir()
                    }
                }
            }
        }

        // Publish into S3 all snapshots versions
        /*stage('Publish snapshot to S3') {
            when {
                branch 'master'
            }
            agent {
                label "linux"
            }
            steps {
                unstash 'linux_whl'
                unstash 'VERSION'
                sh 'echo "Stashed files:" && ls -l dist'
                script {
                    def versionText = utilsLib.getCommandOutput("cat dist/VERSION.txt")
                    def version = utilsLib.fragmentVersion(versionText)
                    def _majorVersion = version[0]
                    def _buildVersion = version[1]
                    version = null // This is necessary, else version:Tuple will be serialized
                    s3up {
                        localArtifact = 'dist/*.whl'
                        artifactId = "h2ogpuml"
                        majorVersion = _majorVersion
                        buildVersion = _buildVersion
                        keepPrivate = true
                    }
                }
            }
        }*/

    }
}
