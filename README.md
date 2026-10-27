# Intelligent Fog–Cloud Distributed Training and Fault-Tolerant MLOps Platform

Plateforme qui répartit l'entraînement d'un modèle de machine learning sur plusieurs machines
physiques (**Fog workers**) selon leurs ressources (CPU, RAM, disponibilité), tandis qu'un
contrôleur central (**Cloud**) pilote l'entraînement, sauvegarde les checkpoints, détecte les
pannes et gère les versions du modèle.

> Projet réalisé par une équipe de 3 étudiants, avec des outils 100 % gratuits et open source.
> Période : **27 octobre 2026 → 2 décembre 2026** (planning v2, machines réelles).

---

## Sommaire

1. [Objectif et idée principale](#1-objectif-et-idée-principale)
2. [Équipe et rôles](#2-équipe-et-rôles)
3. [Architecture](#3-architecture)
4. [Technologies](#4-technologies)
5. [Structure du dépôt](#5-structure-du-dépôt)
6. [Principe de l'entraînement distribué](#6-principe-de-lentraînement-distribué)
7. [Travail à réaliser, phase par phase](#7-travail-à-réaliser-phase-par-phase)
8. [Démarrage](#8-démarrage)
9. [Réseau et ports](#9-réseau-et-ports)
10. [Règles de travail Git](#10-règles-de-travail-git)
11. [Évaluation prévue](#11-évaluation-prévue)
12. [Risques et mesures](#12-risques-et-mesures)
13. [Limites assumées](#13-limites-assumées)

---

## 1. Objectif et idée principale

Au lieu d'entraîner un modèle sur un seul ordinateur, on utilise plusieurs workers qui
entraînent **ensemble** le même modèle. Le contrôleur Cloud :

- surveille le CPU, la RAM et la disponibilité de chaque worker ;
- attribue à chacun une part de données **proportionnelle à ses ressources** ;
- sauvegarde des checkpoints à chaque round ;
- détecte les workers en panne et réattribue leur travail ;
- valide les modèles, gère les versions et permet le rollback ;
- expose un tableau de bord (ressources, progression, pannes).

### Fonctionnalités principales

1. Allocation dynamique de la charge selon les ressources disponibles.
2. Entraînement distribué sur plusieurs Fog nodes.
3. Détection automatique des workers indisponibles ou en panne.
4. Checkpoints et reprise des tâches interrompues.
5. Validation du modèle, gestion des versions et rollback.
6. Dashboard de monitoring (ressources, progression, pannes).

### Résultat attendu

Un prototype fonctionnel qui démontre un entraînement distribué capable de s'adapter à des
ressources changeantes, de survivre à la panne d'un worker et de conserver des versions
fiables du modèle. Comparaison avec deux références (**mono-nœud** et **allocation statique**)
sur : temps d'entraînement, utilisation des ressources, qualité du modèle, temps de récupération.

### Défi central

Coordonner l'entraînement sur plusieurs workers en gardant des mises à jour du modèle
**cohérentes**, et récupérer **proprement** après la panne d'un worker.

---

## 2. Équipe et rôles

| Personne | Rôle | Machine | Responsabilités |
|----------|------|---------|-----------------|
| **A** | Cloud / Backend | Machine 1 (Cloud) | API FastAPI, base PostgreSQL, Celery côté contrôleur, allocation, détection de pannes, reprise |
| **B** | ML / MLOps | Machine 2 (Worker) | Modèle PyTorch, entraînement local, agrégation, checkpoints, MLflow, versions, expériences |
| **C** | Infra / Qualité | Machine 3 (Worker) | Docker, réseau, Prometheus/Grafana, pytest, GitHub Actions, rapport |

**Relecture croisée** : A relit B, B relit C, C relit A.
Chaque membre doit savoir démarrer son worker et **expliquer tout le projet à l'oral**.

---

## 3. Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │          MACHINE 1 : CLOUD (contrôleur)     │
                    │                                             │
   Navigateur ────► │  FastAPI :8000   MLflow :5000               │
                    │  Grafana :3000   Prometheus :9090           │
                    │  Redis :6379 (mot de passe)                 │
                    │  PostgreSQL :5432 (interne uniquement)      │
                    └───────────┬─────────────────┬───────────────┘
                                │                 │
                  Tâches Celery │ + heartbeats    │ upload des poids (HTTP)
                                │                 │
                 ┌──────────────▼───┐       ┌─────▼────────────┐
                 │ MACHINE 2        │       │ MACHINE 3        │
                 │ Fog worker       │       │ Fog worker       │
                 │ PyTorch + Celery │       │ PyTorch + Celery │
                 │ psutil, :8001    │       │ psutil, :8001    │
                 └──────────────────┘       └──────────────────┘
```

### Rôle de chaque composant

| Composant | Rôle |
|-----------|------|
| **FastAPI** | API du contrôleur : enregistrement des workers, jobs, modèles, réception des poids |
| **Redis** | Broker Celery + stockage des heartbeats (clés avec expiration) |
| **Celery** | Envoi des tâches d'entraînement aux workers (une file par worker) |
| **PostgreSQL** | Workers, jobs, événements de panne, backend de MLflow |
| **MLflow** | Suivi des expériences, Model Registry (versions, statut « production ») |
| **Prometheus / Grafana** | Collecte et affichage des métriques de toutes les machines |
| **psutil** | Mesure du CPU et de la RAM disponibles sur chaque worker |

### Flux d'un round d'entraînement

1. Les workers envoient un **heartbeat** (CPU/RAM) toutes les 5 s. Sans heartbeat pendant 15 s, ils sont déclarés **morts**.
2. Le contrôleur calcule la part de données de chaque worker vivant (indices du dataset).
3. Chaque worker charge les poids globaux, entraîne sur sa part, puis **renvoie ses poids** au contrôleur.
4. Le contrôleur fait une **moyenne pondérée** (type FedAvg), sauvegarde un **checkpoint** global et lance le round suivant.
5. Si un worker tombe pendant un round, sa part est **réattribuée** à un worker vivant.
6. En fin d'entraînement, le modèle est évalué ; s'il dépasse le seuil de précision, il est enregistré comme nouvelle version dans MLflow.

---

## 4. Technologies

Python 3.11 · FastAPI · PyTorch (CPU) · Docker Compose · Redis · Celery · PostgreSQL ·
MLflow · Prometheus · Grafana · psutil · pytest · GitHub Actions · Tailscale (réseau privé).

Dataset : **MNIST** (téléchargé localement par chaque worker ; seuls les indices transitent).

---

## 5. Structure du dépôt

```
fog-cloud-mlops/
├── README.md
├── .gitignore
├── .env.example
├── docker-compose.cloud.yml     # (phase 1) Machine 1
├── docker-compose.worker.yml    # (phase 1) Machines 2 et 3
│
├── .github/workflows/           # (phase 9) CI GitHub Actions
│
├── common/        # Code partagé : formats de messages, configuration
├── cloud/         # Machine 1 : API, orchestration, allocation, pannes, MLflow
├── worker/        # Machines 2 et 3 : entraînement local, CPU/RAM, heartbeat
├── ml/            # Modèle PyTorch, baseline mono-nœud, évaluation
├── monitoring/    # Configuration Prometheus et dashboards Grafana
├── scripts/       # Lancement de jobs, expériences, simulation de pannes
├── tests/         # Tests pytest
├── experiments/   # Résultats de la phase 10
└── docs/          # Contrat d'API, fiche des machines, journal, rapport
```

| Dossier | Contenu | Responsable |
|---------|---------|-------------|
| `common/` | Schémas de messages, lecture de la configuration | Tous (via PR) |
| `cloud/` | FastAPI, base de données, orchestrateur de rounds, allocateur, tolérance aux pannes, agrégation, registre de modèles, métriques | A (+ B pour MLOps) |
| `worker/` | Démarrage et enregistrement, tâche `train_round`, mesure des ressources, chargement MNIST, métriques (port 8001) | B |
| `ml/` | `model.py`, `train_single.py`, évaluation | B |
| `monitoring/` | `prometheus.yml`, dashboards Grafana (JSON) | C |
| `scripts/` | Lancement d'un job, expériences, tests de ports, simulation de pannes | C |
| `tests/` | Tests de l'allocation, agrégation, détection de panne, API, registre | Tous |
| `experiments/` | CSV de mesures, courbes, tableaux | Tous |
| `docs/` | `machines.md`, `api-contract.md`, journal d'équipe, architecture, rapport | Tous |

Les fichiers sont créés **au fur et à mesure** des phases ; cette structure peut évoluer.

---

## 6. Principe de l'entraînement distribué

L'entraînement est fait **par rounds** avec une **moyenne pondérée des poids** (approche type
FedAvg / Local SGD). Ce n'est **pas** une synchronisation de gradients (comme PyTorch DDP) :
ce choix rend le système tolérant aux pannes, car la perte d'un worker n'arrête pas le groupe.

- **Allocation statique** (baseline) : parts de données égales.
- **Allocation dynamique** : parts proportionnelles à un score par worker (CPU libre, RAM libre, disponibilité).
- **Agrégation** : poids moyennés, pondérés par le nombre d'exemples traités par chaque worker.
- **Checkpoint** : sauvegardé par le contrôleur à chaque round (pas de volume partagé entre machines).

---

## 7. Travail à réaliser, phase par phase

| Phase | Dates | Objectif | Jalon à valider |
|-------|-------|----------|-----------------|
| 0 | 27 – 28 oct | Installation, dépôt Git, test réseau | Docker, Python, Git OK ; les 3 machines se pingent |
| 1 | 29 oct – 2 nov | Squelette + Compose Cloud et worker | API répond sur la machine 1 ; Redis protégé |
| 2 | 3 – 6 nov | Baseline mono-nœud + MLflow | Temps et précision de référence enregistrés |
| 3A | 7 – 9 nov | Workers + heartbeats en local (3 conteneurs) | Le contrôleur liste les workers vivants |
| 3B | 10 – 11 nov | Déploiement sur les vraies machines | 3 workers sur 3 PC visibles depuis la machine 1 |
| 4 | 12 – 16 nov | Entraînement distribué (allocation statique) | Entraînement complet par rounds sur 3 machines |
| 5 | 17 – 19 nov | Allocation dynamique | Les parts suivent le CPU/RAM réels |
| 6 | 20 – 23 nov | Panne et reprise sur machine réelle | Un worker coupé, l'entraînement se termine |
| 7 | 24 – 25 nov | Validation, versions, rollback | Mauvais modèle refusé ; rollback fonctionnel |
| 8 | 26 – 27 nov | Prometheus + Grafana multi-machines | Dashboard avec les 3 machines et les pannes |
| 9 | 17 – 27 nov (parallèle) | Tests pytest + GitHub Actions | Tests verts dans GitHub Actions |
| 10 | 28 – 29 nov | Expériences de comparaison | Tableaux et courbes mono vs statique vs dynamique |
| 11 | 30 nov – 2 déc | Rapport, diagrammes, démo | Dossier final et démo répétée |

### Détail des tâches par personne

**Phase 0 – Installation et réseau**
- C : créer le dépôt, installer Docker, `docker run hello-world`.
- A, B : installer Docker, Python 3.11, Git ; B teste `import torch`.
- Tous : installer Tailscale, noter les IP, tester le `ping`, désactiver la veille, remplir `docs/machines.md`.

**Phase 1 – Squelette et Compose**
- A : projet FastAPI (`/health`, connexion PostgreSQL, tables `workers` et `training_jobs`).
- C : `docker-compose.cloud.yml` (controller, postgres, redis avec mot de passe), `docker-compose.worker.yml`, `.env`, ouverture des ports.
- B : télécharger MNIST, définir le modèle dans `ml/model.py`.
- Tous : écrire `docs/api-contract.md` (endpoints, formats, envoi des poids par HTTP).

**Phase 2 – Baseline mono-nœud**
- B : `train_single.py` (boucle PyTorch, checkpoint par époque, reprise).
- C : ajouter MLflow au Compose Cloud.
- B + C : enregistrer temps, loss, précision, CPU/RAM ; baseline sur chaque machine (3 exécutions).
- A : endpoint `POST /jobs`.

**Phase 3A / 3B – Workers et heartbeats**
- C : 3 workers en conteneurs sur une machine (limites `cpus`/`mem_limit` différentes), puis déploiement sur les machines 2 et 3.
- A : enregistrement, heartbeat toutes les 5 s, mort après 15 s, `GET /workers`, mesure de latence.
- B : mesure CPU/RAM avec psutil, envoyée avec le heartbeat.

**Phase 4 – Entraînement distribué (statique)**
- A : une file Celery par worker, boucle de rounds, découpage égal des indices.
- B : tâche `train_round`, moyenne pondérée, checkpoint global.
- C : script de lancement d'un job de test, journaux lisibles, vérification de MNIST sur chaque worker.

**Phase 5 – Allocation dynamique**
- A : score par worker et répartition proportionnelle (`allocator.py`).
- B : gestion de parts de taille variable.
- C : tests unitaires de l'allocation, essai avec une machine chargée.

**Phase 6 – Panne et reprise**
- A : réattribution de la part d'un worker mort, limite de tentatives, événement enregistré en base.
- B : reprise du contrôleur depuis le dernier checkpoint.
- C : scénarios (arrêt du conteneur, coupure Wi-Fi, extinction du PC) et mesure du temps de récupération.

**Phase 7 – Validation, versions, rollback**
- B : évaluation sur jeu de validation, enregistrement dans le Model Registry si seuil atteint.
- A : `GET /models`, `POST /models/{version}/rollback`, statut « production ».
- C : tests du refus d'un mauvais modèle et du rollback.

**Phase 8 – Monitoring**
- C : Prometheus (cibles = IP des workers), Grafana, dashboard exporté en JSON.
- A : `/metrics` du contrôleur (workers vivants, round courant, pannes, temps de récupération).
- B : métriques worker sur le port 8001 (CPU, RAM, loss, durée de tâche).

**Phase 9 – Tests et CI (en parallèle)**
- C : workflow GitHub Actions (installation, lint, pytest).
- A, B, C : tests de leurs propres modules. Les tests réseau réels restent manuels.

**Phase 10 – Expériences**
- B : 3 configurations × 3 exécutions (mono-nœud, statique, dynamique).
- C : utilisation des ressources, latence réseau, temps de récupération.
- A : script `scripts/run_experiments.sh`.

**Phase 11 – Rapport et démo**
- A : architecture, allocation, pannes.
- B : modèle, MLOps, résultats, analyse.
- C : diagrammes, captures, scénario de démo.
- Tous : répétition de la démo (10 min) sur les 3 machines, section « Limites ».

---

## 8. Démarrage

### Prérequis (sur chaque machine)

Docker (Desktop avec WSL2 sous Windows, ou Engine + Compose), Git, Python 3.11, Tailscale.

### Configuration

```bash
git clone <url-du-depot>
cd fog-cloud-mlops
cp .env.example .env      # puis modifier les valeurs (CLOUD_IP, mots de passe)
```

Le fichier `.env` contient des secrets : **ne jamais le commiter**.

### Lancement (disponible à partir de la phase 1)

```bash
# Machine 1 (Cloud)
docker compose -f docker-compose.cloud.yml up -d

# Machines 2 et 3 (workers)
docker compose -f docker-compose.worker.yml up -d
```

### Vérification

```bash
curl http://CLOUD_IP:8000/health      # doit répondre « ok »
curl http://CLOUD_IP:8000/workers     # liste des workers vivants (phase 3)
```

### Méthode de développement

1. Développer d'abord avec **3 conteneurs sur une seule machine**.
2. Valider ensuite sur les **vraies machines** : seule l'adresse `CLOUD_IP` change dans `.env`.

---

## 9. Réseau et ports

| Port | Service | Qui se connecte | Exposé |
|------|---------|-----------------|--------|
| 6379 | Redis (mot de passe) | Workers → Cloud | Réseau de l'équipe |
| 8000 | API FastAPI | Workers et utilisateurs → Cloud | Oui |
| 5000 | MLflow | Navigateur → Cloud | Optionnel |
| 3000 / 9090 | Grafana / Prometheus | Navigateur → Cloud | Usage équipe |
| 8001 | Métriques du worker | Prometheus → chaque worker | Sur chaque worker |
| 5432 | PostgreSQL | Services de la machine Cloud | **Non** (interne à Docker) |

Recommandations :
- Utiliser **Tailscale** (IP stables, pas d'isolation Wi-Fi, Redis non exposé sur internet).
- Autoriser les ports dans les pare-feu et tester chaque ouverture (ping + test de port).
- Désactiver la mise en veille des machines pendant les tests.
- Vérifier la synchronisation de l'heure (NTP) sur les 3 machines.

---

## 10. Règles de travail Git

- La branche `main` est protégée : **pas de commit direct**.
- Une branche par tâche : `feat/p1-fastapi-health`, `fix/p3b-firewall`, `docs/p0-machines`.
- Pull Request relue par un coéquipier (A→B→C→A) avant le merge.
- Commits courts, au présent : `feat: add /health endpoint`, `fix: redis password`, `test: add allocator tests`.
- Pas de merge sans test pour l'allocation, l'agrégation et la détection de panne.
- Aucune fonctionnalité n'est « terminée » tant que son jalon n'est pas testé (sur les 3 machines à partir de la phase 3).
- **Journal d'équipe** (`docs/team-log.md`) : date, machine, commande lancée, résultat.
- Point de 15 minutes tous les 2 jours ; les phases 0, 3B et 6 se font avec les 3 membres présents.

---

## 11. Évaluation prévue

| Configuration | Description |
|---------------|-------------|
| Mono-nœud | Entraînement sur une seule machine (référence) |
| Statique | Entraînement distribué, parts égales |
| Dynamique | Entraînement distribué, parts proportionnelles aux ressources |

Chaque configuration est exécutée 3 fois. Métriques : **temps d'entraînement**,
**utilisation des ressources**, **précision du modèle**, **temps de récupération après panne**.
La fiche des machines est jointe aux résultats.

---

## 12. Risques et mesures

| Risque | Mesure |
|--------|--------|
| Wi-Fi qui isole les appareils | Tailscale ou partage de connexion ; test dès la phase 0 |
| Pare-feu qui bloque les ports | Liste des ports ; test à chaque nouvelle machine |
| Adresse IP qui change | IP Tailscale ; `CLOUD_IP` centralisée dans `.env` |
| Machine Cloud éteinte ou en veille | Veille désactivée ; choisir la machine la plus stable |
| Redis exposé | Mot de passe dès la phase 1 ; PostgreSQL jamais exposé |
| Membre absent lors d'un test multi-machines | Planifier les phases 0, 3B et 6 à l'avance |
| Retard cumulé | Sacrifier les extras (dashboard détaillé, rollback avancé), jamais la baseline ni la démo de panne |
| Modèle distribué moins précis que la baseline | Acceptable s'il est mesuré et expliqué |

---

## 13. Limites assumées

- Entraînement par moyenne de poids (type FedAvg), pas de synchronisation de gradients.
- Modèle et dataset volontairement petits (MNIST) pour tenir sur des PC d'étudiants.
- Prototype : pas de sécurité de niveau production, pas de passage à l'échelle testé.
- Résultats dépendants des machines utilisées (voir `docs/machines.md`).
