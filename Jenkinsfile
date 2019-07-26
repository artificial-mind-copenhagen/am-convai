pipeline {

  environment {
    registry = 'dreg.dev.artificialmind.ai'
    registryCreds = 'htpasswdjenkins'
    imageName = 'am-convai'
  }

  agent any

  stages {

    stage('Docker build'){ steps{ script{
      docker.withRegistry('https://dreg.dev.artificialmind.ai', registryCreds) {

        // build container image
        def image = docker.build("${imageName}:${env.BUILD_NUMBER}")

        // push image to registry
        image.push()
      }
    }}}

    stage('Restart on dev') {
      steps {
        sh 'cd /opt/am-deploy && docker-compose up -d ${imageName}'
      }
    }

  } // /stages
}   // /pipeline
