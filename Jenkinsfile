pipeline {
    agent any

    stages {

        stage('Setup') {
            steps {
                echo 'Installing dependencies...'
                bat 'python -m pip install -r requirements.txt'
            }
        }

        stage('Build & Test') {
            steps {
                echo 'Running ML pipeline...'
                bat 'python ml_pipeline.py'
            }
        }
    }

    post {
        success {
            echo 'Pipeline SUCCESS - Model validated'
        }
        failure {
            echo 'Pipeline FAILED - Check logs'
        }
    }
}
