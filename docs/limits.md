## Limites du mode avancé
- Les détecteurs reposent sur des heuristiques simples.
- La classification n’est pas basée sur de l’apprentissage.
- Le mode blind_advanced améliore le réalisme, mais ne garantit pas une reconstruction correcte dans tous les cas.

## Limites du carving
- **JPEG / PNG** : reconstruction par marqueurs (SOI/EOI pour JPEG, signature + footer `IEND` pour PNG).
- **BMP** : le format BMP ne possède **pas de marqueur de fin** (contrairement à JPEG `FF D9` ou PNG `IEND`). La taille du fichier est lue dans le header (uint32 little-endian à l'offset 2) : on découpe donc sur la taille déclarée plutôt qu'en cherchant un footer. La signature `BM` ne fait que 2 octets et apparaît fréquemment par hasard dans un dump ; la **validation du header sert de filtre anti-faux-positifs** — champs réservés nuls (`reserved1 == reserved2 == 0`), taille de header DIB connue (12/40/52/56/64/108/124), `planes == 1`, profondeur `bpp` valide (1/4/8/16/24/32), et cohérence taille/offset des pixels. Un « leurre » `BM` suivi d'octets arbitraires est ainsi rejeté.
- Le carving ne reconstruit pas de fichiers arbitrairement fragmentés à l'octet près (assemblage heuristique par overlap scoring uniquement).