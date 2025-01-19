# Outils de référence en science de données.
# Comment supprimer des conteneurs et des images Docker 🚀

## Supprimer les conteneurs 🛑

1. **Lister les conteneurs en cours d'exécution** :
   ```bash
   docker ps
   ```
   📝 Cette commande affiche uniquement les conteneurs actuellement actifs.

2. **Lister tous les conteneurs (actifs et arrêtés)** :
   ```bash
   docker ps -a
   ```
   📝 Cela montre une liste complète des conteneurs, y compris ceux qui ne sont pas en cours d'exécution.

3. **Obtenir uniquement les IDs des conteneurs** :
   ```bash
   docker ps -a -q
   ```
   📝 Cela renvoie une liste d'IDs des conteneurs sans aucun détail supplémentaire.

4. **Arrêter tous les conteneurs** :
   ```bash
   docker stop $(docker ps -a -q)
   ```
   📝 Cette commande utilise les IDs des conteneurs pour arrêter tous les conteneurs actifs.

5. **Supprimer tous les conteneurs** :
   ```bash
   docker rm $(docker ps -a -q)
   ```
   📝 Après avoir arrêté les conteneurs, cette commande supprime tous les conteneurs listés.

---

## Supprimer les images 🖼️

1. **Lister les images Docker disponibles** :
   ```bash
   docker images
   ```
   📝 Cette commande affiche toutes les images Docker stockées localement avec leurs détails (nom, tag, taille, etc.).

2. **Obtenir uniquement les IDs des images** :
   ```bash
   docker images -q
   ```
   📝 Cela renvoie une liste des IDs des images sans aucun détail supplémentaire.

3. **Supprimer toutes les images Docker** :
   ```bash
   docker rmi $(docker images -q)
   ```
   📝 Cette commande supprime toutes les images en utilisant leurs IDs. Attention : Docker ne supprimera pas les images utilisées par des conteneurs actifs.

---

⚠️ **Attention** : Avant de supprimer, vérifiez que vous n'avez pas besoin des conteneurs ou des images pour éviter toute perte de données ou d'environnement important !


