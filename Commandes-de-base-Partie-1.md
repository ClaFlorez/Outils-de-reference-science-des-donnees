# Commandes de base Docker 🐳

## Liste des commandes et explications ✨

1. **Télécharger une image Docker** :
   ```bash
   docker pull centos
   ```
   📝 Cette commande télécharge l'image officielle `centos` depuis Docker Hub et la stocke localement.

2. **Lister les images Docker disponibles** :
   ```bash
   docker images
   ```
   📝 Cela affiche toutes les images Docker présentes localement avec leurs détails (nom, tag, taille, etc.).

3. **Créer et exécuter un conteneur en arrière-plan** :
   ```bash
   docker run -d --name c1 nginx
   ```
   📝 Cette commande démarre un conteneur nommé `c1` en arrière-plan (`-d`) à partir de l'image `nginx`.

4. **Vérifier les images disponibles** :
   ```bash
   docker images
   ```
   📝 Revoir les images Docker locales après avoir utilisé ou téléchargé une image.

5. **Lister les conteneurs actifs** :
   ```bash
   docker ps
   ```
   📝 Affiche uniquement les conteneurs en cours d'exécution.

6. **Lister tous les conteneurs** :
   ```bash
   docker ps -a
   ```
   📝 Montre tous les conteneurs, y compris ceux qui sont arrêtés.

7. **Arrêter un conteneur actif** :
   ```bash
   docker stop c1
   ```
   📝 Arrête le conteneur nommé `c1` qui était en cours d'exécution.

8. **Supprimer un conteneur** :
   ```bash
   docker rm c1
   ```
   📝 Supprime complètement le conteneur nommé `c1`.

9. **Lister les conteneurs actifs après suppression** :
   ```bash
   docker ps
   ```
   📝 Vérifiez les conteneurs encore actifs après avoir arrêté ou supprimé certains d'entre eux.

10. **Lister tous les conteneurs après suppression** :
    ```bash
    docker ps -a
    ```
    📝 Revoir tous les conteneurs, y compris ceux qui sont arrêtés ou supprimés récemment.

11. **Vérifier les images disponibles après manipulation** :
    ```bash
    docker images
    ```
    📝 Consultez les images locales pour vous assurer qu'aucune image importante n'a été supprimée.

---

💡 **Note** : Ces commandes sont essentielles pour la gestion de base des conteneurs et des images Docker. Elles permettent de manipuler votre environnement Docker simplement et efficacement !

