# Méthodologie : mode assisté vs mode blind

## 1. Mode assisté
Le mode assisté utilise un masque exact ou quasi exact de la zone corrompue.

### Rôle
- valider la cohérence de la chaîne :
  corruption → masque → reconstruction → évaluation
- mesurer la performance maximale du pipeline

### Limite
Ce mode ne doit pas être présenté comme une détection forensic autonome, car la zone à réparer est déjà connue.

---

## 2. Mode blind / réaliste
Le mode blind n’utilise pas directement le masque exact.

### Fonctionnement
- la corruption est appliquée de manière aléatoire
- le masque peut être dégradé ou approximatif
- une détection simple des zones suspectes est utilisée

### Rôle
- simuler un cas plus réaliste
- introduire de l’incertitude
- observer les limites du système

### Limite
La détection repose sur des heuristiques simples, pas sur un modèle avancé de segmentation ou d’IA.

---

## 3. Intérêt expérimental
La coexistence des deux modes permet de distinguer :
- la validation technique du pipeline
- l’évaluation réaliste de la robustesse

Cette distinction est volontaire et doit être explicitée en soutenance.

## Différence entre les modes

### assisted
Masque exact ou quasi exact fourni au pipeline.

### blind_basic
Détection simple basée sur les zones sombres.

### blind_advanced
Détection multi-critères + fusion + classification + stratégie adaptative.