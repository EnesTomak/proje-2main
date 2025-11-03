// Faz 11.2: CI/CD Boru Hattı (Declarative Pipeline)
// Bu dosya, projenin kök dizininde 'Jenkinsfile' olarak kaydedilmelidir.

pipeline {
    agent any // Jenkins'in bu işi herhangi bir uygun ajan üzerinde çalıştırmasına izin ver

    environment {
        // Docker imaj adını tanımla (Docker Hub kullanıcı adınızla değiştirin)
        IMAGE_NAME = "sizin-dockerhub-kullanici-adiniz/proje-2main-web"
        WORKER_IMAGE_NAME = "sizin-dockerhub-kullanici-adiniz/proje-2main-worker"
        IMAGE_TAG = "latest"
        
        // .env dosyasındaki değişkenlerin Jenkins'te 'Credentials' olarak 
        // tanımlanması gerekir. 'google-api-key-creds' adında bir 'Secret text' 
        // credential oluşturduğunuzu varsayıyoruz.
        GOOGLE_API_KEY = credentials('google-api-key-creds')
    }

    stages {
        stage('Checkout') {
            steps {
                // Kodu Git reposundan çek
                git '[httpsSOYADINIZ/proje-2main.git](https://github.com/KULLANICIADINIZ/proje-2main.git)'
            }
        }

        stage('Test (Pytest)') {
            // "Senior" Kanıtı: Kod, production'a gitmeden önce test edilir.
            steps {
                script {
                    echo "Birim testleri (pytest) çalıştırılıyor..."
                    try {
                        // Testleri çalıştırmak için docker-compose'u kullanmak,
                        // 'requirements.txt'deki tüm bağımlılıkların
                        // kurulu olduğu tutarlı bir ortam sağlar.
                        // 'docker-compose.yml'deki 'web' servisini kullanıyoruz.
                        
                        // Önce test ortamını (imajı) build et
                        sh "docker-compose build web"
                        
                        // 'web' servisini (ve bağımlı olduğu 'redis'i)
                        // arka planda (-d) başlat
                        sh "docker-compose up -d web redis"
                        
                        // 'web' container'ı içinde 'pytest' komutunu çalıştır
                        sh "docker-compose exec -T web pytest"
                        
                    } finally {
                        // Testler bittikten sonra (başarılı ya da başarısız)
                        // çalışan container'ları temizle
                        echo "Test ortamı temizleniyor..."
                        sh "docker-compose down"
                    }
                }
            }
        }

        stage('Build Production Images') {
            // Testler başarılı olursa, üretim imajlarını oluştur
            steps {
                script {
                    echo "Docker üretim imajları oluşturuluyor..."
                    
                    // 'docker-compose.yml'deki tüm servisleri build et
                    sh "docker-compose build --no-cache"
                    
                    // İmajları etiketle (tag)
                    // (docker-compose.yml'de 'image:' alanı tanımlıysa bu adıma gerek kalmayabilir,
                    // ancak manuel etiketleme daha net kontrol sağlar.)
                    // sh "docker tag proje-2main_web ${IMAGE_NAME}:${IMAGE_TAG}"
                    // sh "docker tag proje-2main_worker ${WORKER_IMAGE_NAME}:${IMAGE_TAG}"
                }
            }
        }

        stage('Push to Docker Hub (Opsiyonel)') {
            // Bu aşama, imajınızı Docker Hub'a (veya başka bir registry'ye)
            // yükleyerek her yerden erişilebilir hale getirir.
            when {
                // Sadece 'main' branch'ine push yapıldığında çalıştır
                branch 'main' 
            }
            steps {
                // Jenkins'te 'dockerhub-creds' adında bir 'Username with password'
                // kimlik bilgisi (credential) oluşturmanız gerekir.
                withCredentials([usernamePassword(credentialsId: 'dockerhub-creds', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    echo "Docker Hub'a giriş yapılıyor..."
                    sh "docker login -u '${DOCKER_USER}' -p '${DOCKER_PASS}'"
                    
                    echo "İmajlar Docker Hub'a push'lanıyor..."
                    // (docker-compose.yml'de 'image:' alanı tanımlıysa,
                    // 'docker-compose push' komutu yeterlidir)
                    sh "docker-compose push" 
                    
                    // Veya manuel etiketlediyseniz:
                    // sh "docker push ${IMAGE_NAME}:${IMAGE_TAG}"
                    // sh "docker push ${WORKER_IMAGE_NAME}:${IMAGE_TAG}"
                }
            }
        }
    }
    
    post {
        // Pipeline bittikten sonra (başarılı veya başarısız) her zaman çalışır
        always {
            echo 'Pipeline tamamlandı. Temizlik yapılıyor...'
            sh "docker-compose down || true" // Kalan container'ları durdur
            sh "docker logout || true" // Docker Hub'dan çıkış yap
        }
    }
}

