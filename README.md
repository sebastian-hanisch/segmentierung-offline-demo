# Brüche im Rückblick finden: die beste Zerlegung ist exakt, die Zahl der Brüche ist eine Wahl – Streamlit-Demo

<!-- Nach dem Deploy direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-segmentierung-offline-demo.streamlit.app/)** -->

Stück 7 der Reihe **Änderungserkennung** („Wie schnell merkt man, dass sich etwas geändert hat?“) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Stücke 1 bis 6 überwachen eine Reihe **online** und zählen die Lauflänge bis zum Alarm; dieses Stück schaut **im Rückblick** auf eine fertige Reihe und fragt, wo ihr Mittelwert springt. Die Kante zum Vorgänger ist ein `kontrast` des **Maßes** (kein `fix`): bei δ = 1 gibt es keine gemessene Schwäche der Online-Karte, die der Rückblick behebt (der CUSUM liefert die Lage selbst mit, ähnlich gut, Experiment 5); bei δ = 2 ist der Rückblick bei der Lage genauer (gleiche Läufe: Lagefehler 0,53 beim Rückblick gegen 1,56 online, Experiment 5); beide Seiten zahlen Unterschiedliches (online Verzögerung und Fehlalarme, offline die Strafwahl und die ganze Reihe). Plan 06 und Plan 08 nennen „online“ nicht als Schwäche, Plan 08 führt die Kante zu Stück 7 selbst als `kontrast`. Vehikel A: **ausgedachte** Normalreihen mit bekannter Streuung 1 (keine Betriebsdaten). Im Titel und in der App steht „Brüche (Änderungspunkte)“, weil „Segmentierung“ im Bestand auch Welch-Segmente und Mondrian-Klassen meint.

## Warum dieses Problem

Eine Reihe mit Mittelwertsprüngen lässt sich im Rückblick **exakt** in Strecken mit gleichem Mittel zerlegen (Dynamische Programmierung, Optimal Partitioning). Trotzdem hängt das Ergebnis an einer Zahl, die die Mathematik nicht liefert: der **Strafe je Bruch**. **Kernfrage:** (1) Wo liegen die Brüche, und wie viele sind es? (2) Ist die Zahl der Brüche ein Messergebnis oder eine Strafwahl? (3) Findet das schnelle Verfahren (PELT, binäre Segmentierung) dasselbe wie das optimale? (4) Was sagt der Rückblick, was ein Alarm in der Zeit nicht sagt, und umgekehrt?

## Befunde und Korrekturen gegenüber dem Plan

Die Zahlen des Plans (Vormessungen V2 und V16, numpy-Zufall) sind **Erwartungen**. Hier ist alles mit dem SplitMix64-Generator der Demo und eigenen Zufalls-Seeds **neu gemessen** (`tools/sgo_make_fenster.py`, `sgo_make_online.py`, `sgo_make_verfahren.py`; die Fenster-Tabelle aus **drei** Zufalls-Seeds mit je 1000 Läufen, Plan Z5 h). Abweichungen:

| Größe | Plan (Erwartung) | hier gemessen | Einordnung |
|---|---|---|---|
| Nullreihe, n = 200, Mindestlänge 5: Läufe mit Fehlbruch bei c = 1 / c = 2 | 58 bis 62 % / rund 4 % | 59,7 % (Spanne 59,6 bis 59,9) / 3,4 % (2,7 bis 4,0) | im Band |
| Zickzack δ = 1, c = 2: richtige Bruchzahl | 86,8 bis 89,4 % | 86,0 % (Spanne 85,2 bis 87,4) | etwas niedriger; „kein c mit 90 %“ bleibt (in allen drei Zufalls-Seeds) |
| Puls δ = 1, c = 2: richtige Bruchzahl | 88,8 bis 91,5 % | 90,2 % (Spanne 89,0 bis 91,2) | **an der Schwelle**: ein Zufalls-Seed darunter, zwei darüber; die App nennt „knapp“ und die Zahlen |
| Treppe δ = 1, c = 3 | 90,9 und 92,6 % | 91,8 % | im Band |
| Zickzack δ = 1, c = 4 | 18,6 % | 20,1 % | im Band |
| Nullreihe c = 1: Läufe mit Fehlbruch, n = 50 / 100 / 200 / 400 / 800 | 49,9 / 53,5 / 57,7 / 63,2 / 71,2 % | 46,2 / 54,7 / 58,1 / 61,3 / 70,6 % | n = 50 liegt 2,3 Standardfehler eines Anteils (R = 1000) tiefer; Trend gleich |
| Nullreihe c = 2, dieselben n | 6,2 / 5,3 / 3,7 / 2,8 / 1,0 % | 6,3 / 4,2 / 4,1 / 2,1 / 1,6 % | im Band |
| Mindestlänge 1 / 2 / 5 / 10 / 20, Nullreihe c = 1: Fehlbruch | 78,8 / 72,2 / 59,1 / 45,2 / 32,5 % | 80,0 / 73,3 / 59,5 / 47,3 / 34,3 % | im Band |
| Kurzer Puls (δ = 3, c = 2, Mindestlänge 5), Breite 2 / 4 / 5: richtige Zahl | 8,2 / 84,4 / 94,8 % | 12,0 / 83,8 / 94,5 % | Breite 2: 3,8 Prozentpunkte höher (3,7 Standardfehler eines Anteils bei R = 1000); Aussage „fällt weg“ bleibt |
| PELT ≠ optimal, n = 200, Mindestlänge 5 (acht Zufalls-Seeds) | 5 von 8 320 | 2 von 8 320 | beide „sehr selten“; die Aussage der App ist „nur bei Mindestlänge 1 beweisbar exakt“ |
| PELT ≠ optimal, zufällige ganzzahlige Reihen n ≤ 16, Mindestlänge 2 / 3 / 5 (je 1500) | 66 / 51 / 32 | 58 / 47 / 38 | anderer Generator, andere Zahlen: nur der Befund „> 0“ gilt; generatorfrei sind die 40 und 14 von 1024 für n = 5 (unverändert) |
| Binäre Segmentierung ≠ optimal, exakt n = 4 bis 14 (3000 Instanzen) | 519 (Zielwert größer in 498, kleiner in 0) | 508 (größer in 480, kleiner in 0) | anderer Generator |
| Binäre Segmentierung ≠ optimal, n = 200, 13 Zellen × c = 1, 2, 4 | 38,7 % (55,0 / 34,4 / 26,8 % je c) | 39,2 % (55,2 / 35,2 / 27,2 %) | im Band; ein Bruch, c = 2: 1,8 / 1,0 / 1,0 % für δ = 1 / 2 / 3 (Plan: höchstens 1,2 % bei δ ≥ 2) |
| Kostenauswertungen PELT / DP (Mindestlänge 1 / 5), verzögert (5) | 0,355 / 0,361, 0,388 | 0,361 / 0,380, 0,407 | **andere Zählung:** die DP zählt hier nur zulässige Strecken (Startpunkt 0 oder ≥ Mindestlänge); die Vormessung zählte alle Startpunkte |
| CUSUM, ν = 100, R = 4000: Alarm vor ν (Simulation) / Verzögerung δ = 1 / Lagefehler online / offline | 23,6 % / 8,010 ± 0,090 / 2,64 / 3,34 | 22,7 % / 7,885 ± 0,085 / 2,57 / 3,14 | exakt unverändert (23,02 %, 7,9058); Lagefehler im Band |
| Rückblick, genau ein Bruch (δ = 1 / 2) | 95,3 / 95,0 % | 95,6 / 95,3 % | im Band |
| Phase I, N = 100, Sprung in der Mitte, δ = 0 / 0,5 / 1 | 4,8 / 49,5 / 98,9 % | 5,9 / 48,9 / 98,7 % | δ = 0: 2,4 Standardfehler höher (Standardfehler eines Anteils bei R = 2000) |
| Streuung unbekannt (neu gemessen, Z5 f): Zickzack δ = 1, c = 2 | 84,0 gegen 88,5 % | 81,2 gegen 86,0 % | Kosten der robusten Schätzung 0,8 bis 4,7 Prozentpunkte je Zelle, am größten bei Zickzack und Puls δ = 1 (beide im Rauschen ähnlich) |

Weitere Korrekturen beim Bau:

- **PELT bei β = 0 (Fund des Randfall-Tests):** Die Beschneidungsregel F(s) + C(s, t) ≥ F(t) streicht bei β = 0 den besten Kandidaten selbst (dort ist F(t) = F(s) + C(s, t)); bei Mindestlänge > 1 bleibt die Kandidatenliste leer und die Rechnung bricht ab. β ≤ 0 ist kein Fall der Quelle (Killick u. a. setzen eine positive Strafe voraus), aber der Regler erlaubt β = 0. Gebaut: bei β ≤ 0 gilt die strenge Ungleichung; für β > 0 ist nichts anders (Handbeispiele und Orakel unverändert).
- **Zufallsquelle (Plan Z3):** Der Plan sah die Klasse der Vorlage vor, deren Normalzahlen 61 bis 131 µs je Zahl brauchen. Gebaut ist die **Stapel-Fassung desselben SplitMix64** (`sgo_normal.py`, bitgleich mit `sgo_rng.py`, `tests/test_rng.py`): Normalzahlen über `scipy.special.ndtri`. Folge: **scipy ist Laufzeit-Bibliothek** (der Plan sagte „numpy und stdlib“; `ndtri`, `ndtr` und `brentq`), und die Obergrenzen des Plans (live R ≤ 200, n · R ≤ 40 000) sind sehr konservativ (siehe Rechenzeiten); sie bleiben, weil sie Streamlit Cloud schonen, dort ist nichts gemessen.
- **Lagen der wahren Brüche** sind Anteile von n (0,5 / 0,35, 0,7 / 0,4, 0,6 / 0,25, 0,5, 0,75); bei n = 200 sind es die Lagen der Vormessung (100 / 70, 140 / 80, 120 / 50, 100, 150).
- **Schnellstarts:** acht statt sieben; neu ist „Streuung unbekannt: Zickzack δ = 1“ (Plan Z5 f verlangt den Schalter). „Online gegen offline (δ = 1)“ unterscheidet sich von „Ein Bruch, schwach (δ = 1)“ durch 200 statt 100 Live-Läufe.
- **Gesperrte statt ausgeblendete Regler:** Strafe als c · ln n oder absolut, eigene Reihe oder Szenario: der jeweils unwirksame Regler ist gesperrt (`disabled`), nicht ausgeblendet; so entsteht kein Regler, der still wirkungslos bleibt.
- **Experiment 5 mit gleichen Läufen (Z5 g):** zusätzlich zu „alle Läufe“ steht die Auswertung der Läufe **ohne früheren Alarm** (3093 von 4000 bei ν = 100, δ = 1); die Lagefehler beider Seiten sind dort auf derselben Stichprobe gerechnet (δ = 1: online 2,57, offline 2,78 bei genau einem Bruch; δ = 2: online 1,56, offline 0,53).
- **Experiment 3** nutzt Mindestlänge 1 bis n = 10 (dort gilt die exakte Rechnung für n = 2 und 3) und 5 ab n = 50; das ist eine Festlegung dieser Demo.
- **Tab „Toleranz“** bleibt (Plan: erster Kandidat zum Streichen bei Überlänge); die Seite ist lang, aber nicht geteilt.

## Modell

Reihe x₁ … xₙ mit stückweise konstantem Mittelwert und Normalrauschen mit **bekannter Streuung 1**, unabhängig. **Bruch** s heißt: s Beobachtungen liegen vor der Änderung (ν-Konvention). **Kosten** einer Strecke = Summe der Abweichungsquadrate um ihr Mittel; **Zielwert** = Summe der Kosten + β · Zahl der Brüche; **Mindestlänge** m je Strecke. Strafe β = c · ln n (**c = 2 heißt „BIC-Wahl“: eine Setzung dieses Berichts, keine Quellenaussage**). **Eigenes Maß, keine ARL:** gemessen werden Bruchzahl, Bruchlage (Fehler in Beobachtungen), Zielwert und F1 mit Toleranz M. Zähleinheiten: **Läufe** (eine simulierte Reihe), **Zerlegungen** (2ⁿ⁻¹ bei m = 1, Fibonacci-Zahlen bei m = 2), **Kostenauswertungen** (zulässige Strecken), Beobachtungen; „Fehlbruch“ (gefundener Bruch ohne wahren Bruch, Läufe gezählt) und „Fehlalarm“ (nur Alarm der Online-Karte vor ν, ARL₀) sind getrennt. Die Verzögerungsmaße (Lorden, Pollak, Shiryaev) gelten für die Zerlegung nicht; in Experiment 5 steht die Verzögerung bei **fester Lage ν, gegeben kein Alarm vor ν** (zero-state-Start, ν Schritte Einschwingen, für den CUSUM praktisch das steady-state-Maß; nicht das Lorden- und nicht das Shiryaev-Maß); das ARL₀-Ziel ist **370,398** (nicht das gesetzte 370). Die Formeln stehen im 📐-Expander der App (vollständiges Modell).

## Methodik

- **Optimale Zerlegung:** Optimal Partitioning ohne Abkürzung, vektorisiert über Läufe und Strafen; jede Reihe wird vorher **zentriert** (Verschiebung um 10⁸ ändert die Brüche nicht, Test). Gleichstand: kleinster Startpunkt der letzten Strecke. Zweiter Rechenweg in den Tests: Aufzählung aller Zerlegungen und die Fraction-DP.
- **PELT** mit K = 0 und das **verzögerte Streichen** (eigener Vorschlag, keine Quellenaussage; Beweis für den Zielwert im 📐-Expander, Bausteine im Test), **binäre Segmentierung** (Teilung nur bei Gewinn > β).
- **Nullreihe:** n = 2 geschlossen (χ²₁), n = 3 als Winkelintegral (Gauß-Legendre, Knickstellen bei 15°, 30°, 45°, 105°, 120°, 135°), beide gegen ein zweites Integral und gegen Simulation.
- **Online-Seite:** einseitiger CUSUM, k = 0,5, h für ARL₀ = 370,398 aus der Integralgleichung (48 Knoten), gegen die Brook/Evans-Kette (400 und 800 Zellen); die Verzögerung und die Wahrscheinlichkeit eines Alarms vor ν exakt aus der Kette.
- **Gefrorene Tabellen** (`sgo_fenster.json`, `sgo_null_n.json`, `sgo_online.json`, `sgo_verfahren.json`): jede Zelle mit Zufalls-Seed und R, Kopf mit Quelle und Stand; `tools/sgo_make_*.py --voll` rechnet neu (zusammen rund 3 Minuten), `--check` prüft im Band (wenige Sekunden). Simulationsanteile sind **gelb** (Standardfehler eines Anteils bis 1,6 Prozentpunkte bei R = 1000), exakte Teile **grün**.
- **Live-Rechenzeiten** (diese Maschine, ein Thread, kein Befund): eine Reihe mit n = 400 und allen vier Verfahren 0,013 s; Experiment 2 mit R = 200, n = 200 und sieben Strafen 0,08 s; Experiment 3 mit n = 800, R = 50 0,05 s; Experiment 4 mit R = 100 0,45 bis 0,48 s; Experiment 5 mit R = 200 0,03 s. Streamlit Cloud ist nicht gemessen.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen gelten für: Mittelwertsprünge, Normalrauschen mit bekannter Streuung 1, unabhängig, ausgedachte Reihen, optimale Zerlegung mit Mindestlänge 5 und n = 200, wo nicht anders genannt; Zähleinheit **Läufe** (R = 1000 je Zufalls-Seed, drei Zufalls-Seeds, Mittel der drei) bzw. Instanzen, Zerlegungen, Kostenauswertungen. „Richtige Bruchzahl“ heißt: genau so viele Brüche wie wahre.

| Frage | Befund | Test (Datei, Funktion) |
|---|---|---|
| Handbeispiel 1: F(0 bis 6) für (0 0 0 2 2 2), β = 2 | −2, 0, 0, 0, 2, 2, 2; ein Bruch bei 3, Zielwert 2 (nur 1 von 32 Zerlegungen) | `test_handbeispiele.py::test_h1_*` |
| Handbeispiel 4: Puls (0 0 0 2 2 2 0 0 0) | Kosten 8, größter Einzelgewinn 2; für 2 ≤ β < 4 findet die DP 3 und 6, die binäre Segmentierung nichts | `test_handbeispiele.py::test_h4_*` |
| Handbeispiel 5: (2 1 0 0 2), Mindestlänge 2, β = 1 | optimal 4 ohne Bruch; PELT 25/6 mit Bruch bei 2; verzögert 4 | `test_handbeispiele.py::test_h5_gegenbeispiel_*` |
| Alle 1024 ganzzahligen Reihen mit n = 5, Werte 0 bis 3, Strafe 1 und 2 (2048 Instanzen) | Mindestlänge 1: PELT weicht in 0 ab; Mindestlänge 2: in 40 (β = 1) und 14 (β = 2) von 1024; verzögert 0 | `test_handbeispiele.py::test_h5_alle_1024_*`, `test_tables.py::test_verfahren_exakte_zaehlungen_n5` |
| Nullreihe n = 2: Wahrscheinlichkeit eines Fehlbruchs bei c = 0,5 / 1 / 2 / 3 (exakt) | 55,606 / 40,510 / 23,903 / 14,929 % | `test_oracle_nullreihe.py::test_n2_*` |
| Nullreihe n = 3: P(kein Bruch) bei c = 0,5 / 1 / 2 / 3 (Winkelintegral, geprüft gegen ein zweites Integral) | 0,304008 / 0,511109 / 0,752975 / 0,871957 | `test_oracle_nullreihe.py::test_n3_*` |
| Feste Stelle bei n = 200, c = 1 / 2: 2·Φ̄(√β); Zahl der Stellen (Mindestlänge 5) mal Einzelwert | 2,13 % / 0,113 %; 191 Stellen: 408 % / 21,6 % (**keine Schranke**) | `test_handbeispiele.py::test_h6_feste_stelle_*` |
| Nullreihe n = 200: Läufe mit mindestens einem Fehlbruch bei c = 1 / 2 / 3 | 59,7 / 3,4 / 0,2 % | `test_tables.py::test_fenster_aussagen_*` (Band), `test_claims.py::test_fenster_nullreihe` |
| Nullreihe: mittlere Zahl gefundener Brüche bei c = 0,5 / 1 / 2 | 9,01 / 1,54 / 0,04 | `test_claims.py::test_fenster_nullreihe` |
| Nullreihe c = 1: Läufe mit Fehlbruch, n = 50 / 100 / 200 / 400 / 800 (R = 1000) | 46,2 / 54,7 / 58,1 / 61,3 / 70,6 % (steigt); c = 2: 6,3 / 4,2 / 4,1 / 2,1 / 1,6 % (fällt); belastbar ist der Trend, die Einzelwerte haben bei R = 1000 und einem Zufalls-Seed etwa 1,5 Prozentpunkte Standardfehler | `test_claims.py::test_nullreihe_gegen_n` |
| Ein Bruch δ = 1: richtige Bruchzahl bei c = 1 / 2 / 3 / 4 / 6 / 10 | 35,7 / 95,7 / 99,7 / 99,4 / 94,8 / 45,1 %; Fenster (≥ 90 %): c = 2 bis 6 | `test_claims.py::test_fenster_ein_bruch` |
| Treppe δ = 1: richtige Bruchzahl bei c = 2 / 3 / 4 | 93,3 / 91,8 / 76,0 %; Fenster c = 2 bis 3 | `test_claims.py::test_fenster_treppe_puls_zickzack` |
| Puls δ = 1: richtige Bruchzahl bei c = 2 / 3 / 4 | 90,2 % (Spanne der drei Zufalls-Seeds 89,0 bis 91,2: **an der Schwelle**) / 68,5 / 33,4 % | `test_claims.py::test_fenster_treppe_puls_zickzack` |
| Zickzack δ = 1: richtige Bruchzahl bei c = 2 / 3 / 4 / 6 | 86,0 % (85,2 bis 87,4) / 56,7 / 20,1 / 0,4 %; **kein c mit 90 %** (auch nicht in der Spanne; n = 200, Mindestlänge 5, bekannte Streuung) | `test_claims.py::test_fenster_treppe_puls_zickzack`, `test_tables.py::test_fenster_aussagen_*` |
| Zickzack δ = 1: Läufe mit genau der wahren Bruchmenge bei c = 2; mittlere Zahl gefundener Brüche | 1,7 % (1,4 bis 1,9); 2,89 statt 3 | `test_claims.py::test_toleranz_f1` |
| Fenster bei δ = 2 und 3 (alle vier Szenarien, Zufalls-Seeds gemittelt) | ein Bruch, Treppe δ = 2, 3 und Puls, Zickzack δ = 3: c = 2 bis 10; Puls und Zickzack δ = 2: c = 2 bis 6 | `test_claims.py::test_fenster_delta_2_3` |
| Lagefehler bei richtiger Zahl, c = 2: ein Bruch δ = 1 / 2 / 3 (Beobachtungen) | 3,27 / 0,59 / 0,15 (Mittel mit langem Tail: die zweite Stelle hängt am Zufalls-Seed, eine Nachrechnung mit 8000 Läufen gab 3,12 / 0,63 / 0,18) | `test_claims.py::test_lagefehler` |
| F1 mit Toleranz M = 1 / 2 / 5 / 10 / 20, ein Bruch δ = 1, c = 2 | 0,274 / 0,477 / 0,755 / 0,902 / 0,970 | `test_claims.py::test_toleranz_f1` |
| F1 mit M = 1 / 2 / 5 / 10 / 20, Zickzack δ = 1 | 0,249 / 0,458 / 0,719 / 0,858 / 0,932 (zulässig nur M < 25, eigene Verschärfung) | `test_claims.py::test_toleranz_f1` |
| Streuung unbekannt (robuste Schätzung), c = 2: Zickzack δ = 1 / Puls δ = 1 / ein Bruch δ = 1: richtige Bruchzahl gegen bekannte Streuung | 81,2 gegen 86,0 / 85,9 gegen 90,2 / 93,3 gegen 95,7 % | `test_claims.py::test_streuung_unbekannt` |
| Mindestlänge 1 / 2 / 5 / 10 / 20: Nullreihe mit Fehlbruch bei c = 1 / 2 / 3 | 80,0 / 4,5 / 0,4 · 73,3 / 4,4 / 0,4 · 59,5 / 3,5 / 0,4 · 47,3 / 3,1 / 0,4 · 34,3 / 2,7 / 0,3 % (R = 1000, ein Zufalls-Seed) | `test_claims.py::test_mindestlaenge` |
| Kurzer Puls (δ = 3, c = 2, Mindestlänge 5, R = 1000): richtige Zahl bei Breite 2 / 4 / 5 / 8 / 20 | 12,0 / 83,8 / 94,5 / 95,8 / 96,0 % | `test_claims.py::test_mindestlaenge` |
| PELT ≠ optimal bei n = 200 (13 Zellen × 40 Läufe × c = 1, 2 je Zufalls-Seed): Mindestlänge 1 (ein Zufalls-Seed) / 5 (acht) / 2 (zwei) | 0 von 1040 / 2 von 8320 / 0 von 2080; verzögert: 0 in allen | `test_claims.py::test_pelt_n200` |
| Zufällige ganzzahlige Reihen n ≤ 16 (je 1500, Zufallsquelle der Demo): PELT ≠ optimal bei Mindestlänge 1 / 2 / 3 / 5 | 0 / 58 / 47 / 38; verzögert 0 in allen | `test_claims.py::test_pelt_zufallsreihen_exakt` |
| Kostenauswertungen im Verhältnis zur DP (Mindestlänge 1 / 5) | PELT 0,361 / 0,380; verzögert (5) 0,407 | `test_claims.py::test_pelt_n200` |
| Binäre Segmentierung gegen DP, exakt, n = 4 bis 14 (3000 Instanzen) | andere Bruchmenge in 508, Zielwert größer in 480, kleiner in 0 | `test_claims.py::test_binaere_segmentierung` |
| Binäre Segmentierung, n = 200, 13 Zellen × c = 1, 2, 4 (400 Läufe je Zelle) | andere Bruchmenge in 6112 von 15 600 (39,2 %); je c 55,2 / 35,2 / 27,2 %; Zielwert nie besser; je Zelle bei c = 2 von 0,5 bis 77,8 % | `test_claims.py::test_binaere_segmentierung` |
| CUSUM k = 0,5, ARL₀ = 370,398: Schwelle h; ARL₀ der Kette (400 / 800 Zellen) | h = 4,09649; 370,3887 / 370,3959 | `test_oracle_online.py`, `test_claims.py::test_online_exakt` |
| ARL₁ (zero-state) δ = 1 / 2 (Integralgleichung und Kette) | 8,5751 / 3,4068 auf beiden Wegen | `test_oracle_online.py::test_arl1_*` |
| CUSUM, ν = 100: Alarm vor ν (exakt); Verzögerung δ = 1 / 2 gegeben kein Alarm vor ν (exakt) | 23,02 %; 7,9058 / 3,1090 Beobachtungen | `test_oracle_online.py`, `test_claims.py::test_online_exakt` |
| Rückblick (c = 2, Mindestlänge 5, 4000 Läufe): genau ein Bruch δ = 1 / 2; Lagefehler bei genau einem Bruch | 95,6 / 95,3 %; 3,14 / 0,56 (Standardfehler etwa 0,1 und 0,02; eine Nachrechnung mit 60 000 Läufen gab 3,20 / 0,62) | `test_claims.py::test_online_simulation` |
| Gleiche Läufe (3093 von 4000, kein Alarm vor ν), δ = 1: Lagefehler online / offline | 2,57 / 2,78 | `test_claims.py::test_online_simulation` |
| Gleiche Läufe (3093 von 4000, kein Alarm vor ν), δ = 2: Lagefehler online / offline | 1,56 / 0,53 | `test_claims.py::test_online_simulation` |
| Phase I, N = 100, ein verdeckter Sprung, c = 2: gefunden bei δ = 0 / 0,5 / 1 / 1,5 / 2, Sprung in der Mitte (R = 2000) | 5,9 / 48,9 / 98,7 / 100 / 100 %; am Rand (bei 10): 5,3 / 17,2 / 63,5 / 96,2 / 99,8 %; bei δ = 0 etwa 5 % (4,8 bis 5,9) | `test_claims.py::test_phase1` |
| Phase I, Sprung δ = 1 in der Mitte von N = 100: Mittel der Stichprobe und Streuung (statt 0 und 1) | 0,5 und 1,118 (= √(1 + δ²/4)) | `test_claims.py::test_phase1` |
| 370,398 = 1/(2 Φ(−3)); die Zahl der Zerlegungen von n = 6 mit Mindestlänge 2 | exakt; 5 | `test_claims.py::test_ziel_und_zerlegungen` |
| Live-Läufe unter 1 s (Messung dieser Maschine, kein Befund); der Test begrenzt nur nach oben (60 s) | Messung dieser README, nicht im Test | `test_edge_cases.py::test_worst_case_am_regler_maximum_mit_zeitgrenze` |

**Gelesen als Zählung:** „Die Strafe ist ein Fenster“ gilt für die **13 Zellen** (ein Bruch, Treppe, Puls, Zickzack je δ = 1, 2, 3, plus Nullreihe) bei n = 200, Mindestlänge 5, bekannter Streuung und die sieben Werte c = 0,5 bis 10: in **elf bis zwölf von dreizehn** Zellen gibt es ein Fenster (die Nullreihe hat „kein Bruch“ als richtige Zahl: c = 2 bis 10), **nicht beim Zickzack δ = 1**; beim Puls δ = 1 steht es auf der Schwelle, dort hängt die Zählung vom Zufalls-Seed ab. „Die untere Kante ist scharf“: Nullreihe c = 0,5 / 1 / 2 gibt 9,01 / 1,54 / 0,04 Brüche je Lauf. „Die obere Kante hängt an Sprunghöhe und Bruchzahl“: Zickzack δ = 1 bei c = 4 / 6: 20,1 / 0,4 %, bei δ = 3 in diesem Raster mindestens 99,7 % für c = 3 bis 10. „PELT ist bei Mindestlänge 1 exakt“ gilt als **Beweis** (Killick u. a., Satz 3.1, Abstract und Abschnitte 1 bis 4 gelesen) und als Zählung in 0 von 2048 (n = 5), 0 von 1500 (n ≤ 16) und 0 von 1040 Instanzen (n = 200); „PELT weicht bei Mindestlänge ≥ 2 ab“ gilt **für diese Implementierung der Regel mit K = 0**, ob Satz 3.1 die Mindestlänge ausschließt, ist nicht geprüft. „Das verzögerte Streichen gleicht dem Optimum“ gilt für den **Zielwert** (Beweis, 📐) und als Zählung in 0 von 6548 exakten Instanzen mit Mindestlänge ≥ 2 (2048 mit n = 5, dazu 4500 zufällige Reihen n ≤ 16 mit Mindestlänge 2, 3 und 5) und 0 von 10 400 Gleitkomma-Instanzen mit n = 200 (Mindestlänge 5 und 2); bei Gleichständen kann die Bruchmenge abweichen. „Die binäre Segmentierung ist nie besser“: in 3000 exakten und 15 600 Gleitkomma-Läufen.

## Ehrliche Grenzen und 🚧 Nicht gemessen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Mittelwertsprünge, Streuung bekannt, unabhängiges Normalrauschen** | Autokorrelation und Schiefe (Romano u. a. 2022: Standardstrafen überschätzen die Bruchzahl bei autokorreliertem Rauschen) sind **nicht gemessen**; die Gate-Reihe (4,46 Fehlbrüche bei c = 2, Planung V2) und die Residuen des `niveausprung-demo`-Erzeugers sind **nicht gemessen** | Stück 5, Stück 8 |
| **Ausgedachte Reihen** | In den gelesenen Quellen gibt es keine reale Datenquelle für diese Linie; was ein Rückblick-Verfahren oder eine Alarmregel in Logistik, Lager oder Fertigung nützt, wird hier **nicht behauptet** | – |
| **Eine Reihe, Mittelwertsprung** | Varianzwechsel, mehrere Kanäle, Wild Binary Segmentation, BOCPD, nichtparametrische Kosten: nicht gemessen | offen |
| **PELT mit K = 0, Mindestlänge** | Gilt für diese Implementierung; ob Killick u. a. die Mindestlänge ausschließen und wie `ruptures` damit umgeht, ist **nicht geprüft** (Paket nicht installiert, Wortlaut nur Abstract und Abschnitte 1 bis 4) | offen |
| **Phase I als Offline-Suche** | Die Folge für die Karte (bedingte ARL₀ nach einer Phase I mit verdecktem Sprung) ist **nicht gemessen** | Stück 4 |
| **Online-Seite** | Einseitiger CUSUM nach oben; Lorden- und Shiryaev-Verzögerung sind nicht gemessen; der Rückblick unterscheidet die Richtung nicht | Stück 2 |
| **Van den Burg/Williams** | Ihr Befund, dass im Standard kein Verfahren signifikant besser ist als „kein Änderungspunkt“, gilt auf **ihren** 37 realen Reihen und ist hier nicht nachgemessen; die „Null-Methode“ hat hier F1 = 0 (Festlegung), welche Konvention sie benutzen, ist im gelesenen Auszug nicht geprüft | – |
| **Kostenauswertungen** | Ein Zählmaß, keine Laufzeit; Sekunden sind kein Befund | – |
| **Rechenumgebung** | Gefrorene Zahlen unter Windows mit Python 3.14 gerechnet; Normalzahlen (Umkehrfunktion) über Plattformen nur bis etwa 1e-15 gleich, Tests prüfen Bänder; Linux-Lauf (`demo_linux_check.py`, Docker) **nicht** ausgeführt; Streamlit-Cloud-Zeiten nicht gemessen | – |
| **Nutzen** | Der Nutzen in Betrieb ist **nicht belegt** | – |

**Quellen und Stufen.** Killick/Fearnhead/Eckley 2012 (arXiv 1101.1438) **gelesen (Stufe W)**, Abstract und Abschnitte 1 bis 4: PELT löst die bestrafte Segmentierung exakt mit Beschneidung, linear nur unter Bedingungen. Truong/Oudre/Vayatis 2020 **gelesen (W)**, Abschnitte 2, 3.2, 5.1: Kosten × Suche × Nebenbedingung; F1 mit Toleranz nur sinnvoll, wenn M kleiner als der kleinste Abstand wahrer Brüche ist. Van den Burg/Williams **gelesen (W)**. Romano u. a. 2022 **gelesen (W)**, Abstract und Einleitung. Bai/Perron **gelesen (W)**, nur die Arbeitsfassung CIREQ 9807. Fryzlewicz (WBS) und Adams/MacKay (BOCPD) gelesen, hier **nicht gemessen**. **Nicht gelesen:** Basseville/Nikiforov, Page 1954, Jackson u. a. 2005, Scott/Knott 1974 im Original. **Keine Neuheit:** alles ist Lehrbuch und Stand der Technik; neu ist die Vorführbarkeit mit Zähleinheit und exakten Orakeln.

## Tests

**249 Tests**, seriell rund 136 s (diese Maschine unter Last anderer Sitzungen, Windows, Python 3.14, ein BLAS-Thread; `pytest tests/ -q`). Zusammensetzung:

- **Handbeispiele** (`test_handbeispiele.py`): F-Tabelle, Puls, PELT-Gegenbeispiel (auch alle 1024 Reihen), Zerlegungszahlen, Nullverteilung, CUSUM-Folge, F1; Brüche und Ganzzahlen von Hand.
- **Orakel** (`test_oracle_segmentierung_offline.py`, `test_oracle_nullreihe.py`, `test_oracle_online.py`): DP gegen Aufzählung (exakt, `fractions`), Gleitkomma-DP gegen exakt (auch nicht ganzzahlige Reihen), Aufzählung der Demo gegen die exakte, PELT und verzögertes Streichen gegen DP, binäre Segmentierung nie besser; n = 2 und 3 auf zwei Wegen und gegen Simulation (Band 4 Standardfehler); Schwelle und ARL der Kette gegen die Integralgleichung, exakte Verzögerung gegen Simulation.
- **Eigenschaften** (`test_properties.py`): Verschiebung bis 10¹⁰, Skalierung, Spiegelung, Monotonie in β und Mindestlänge, β = 0, Gewinnformel, Gleichstand.
- **Randfälle** (`test_edge_cases.py`): leere und nicht numerische Eingabe, Dezimalkomma, n = 1, 2, Mindestlänge über n, n = 14 und 15 an der Aufzählungsgrenze, Obergrenzen, R = 1, Regler am Maximum (Zeitgrenze).
- **Tabellen** (`test_tables.py`): Aufbau, Zufalls-Seed und R je Zelle, Abgleich mit den Vormessungen im Band, exakte Teile gleich Neurechnung, `--check` der drei Werkzeuge.
- **Claims** (`test_claims.py`): jede Zahl der Tabelle oben, neu aus den Tabellen erzeugt und wörtlich in der README gesucht; Wortliste.
- **Verdikte** (`test_verdicts.py`: reine Funktionen an von Hand gesetzten Zahlen; `test_preset_verdicts.py`: je Schnellstart die erwartete Zeile), **Layout** (`test_layout.py`: Schnellstart in der Seitenleiste, fester Kopf, Hervorhebung, jeder Regler hebt sie auf), **App** (`test_app.py` Smoke-Vorlage, `test_app_experimente.py`: Verdikte im Standard, jeder Schnellstart, jede Option, gesperrte Regler, Fehlermeldungen, Obergrenze), Schnellstarts, Abbildungen, Footer, Laufzeit-Importe, Zufallsquelle.
- **Plattform:** Neu-Rechnen-Tests sind Bänder, nie bitgleich; Gleitkomma-Letztstellen nur mit Toleranz; die Gleichstand-Auswahl der Gleitkomma-DP ist nicht bitgleich mit der exakten (verglichen wird der Zielwert).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore`, `.streamlit/config.toml` | venv, Caches, Secrets; keine Nutzungsstatistik |
| `app.py` | Streamlit-App (fünf Experimente, Kernfragen, 🚧, 📐) |
| `sgo_segmentation.py` | Kern: optimale Zerlegung, PELT (auch verzögert), binäre Segmentierung, Aufzählung, F1 |
| `sgo_exact.py`, `sgo_online.py` | exakte Nullverteilung n = 2, 3; CUSUM, Kette, Integralgleichung |
| `sgo_scenario.py` | Szenarien, Eingabe der eigenen Reihe, Live-Läufe der fünf Experimente |
| `sgo_normal.py`, `sgo_rng.py` | Stapel-Normalzahlen aus SplitMix64; Vorlage `sgo_rng.py` unverändert |
| `sgo_reference.py` und `sgo_fenster.json`, `sgo_null_n.json`, `sgo_online.json`, `sgo_verfahren.json` | gefrorene Tabellen (Seed und R je Zelle) und Zugriff |
| `sgo_evaluation.py`, `sgo_texts.py` | Einstellungen, Verdikte (reine Funktionen), Texte |
| `sgo_constants.py`, `sgo_presets.py`, `sgo_layout.py`, `sgo_visualization.py` | Regler und Schnellstarts, Permalink, fester Kopf, Plotly-Abbildungen |
| `tools/sgo_make_fenster.py`, `tools/sgo_make_online.py`, `tools/sgo_make_verfahren.py` | erzeugen (`--voll`) und prüfen (`--check`) die Tabellen |
| `tests/` | siehe Abschnitt Tests; `tests/exact_ref.py` ist die exakte Referenz (Testhilfe) |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit (streamlit, numpy, scipy, pandas, plotly), dazu pytest |

## Bewusst nicht umgesetzt

- **Autokorrelation, Gate-Reihe, Residuen des Niveausprung-Erzeugers:** eine eigene Frage (Stück 5 und 8).
- **Varianzwechsel, mehrere Kanäle, WBS, BOCPD, nichtparametrische Kosten.**
- **Kein Nachbau** von `niveausprung-demo` (Anwendung: Residuen, Zurücksetzen des Prognosemodells, Bestätigungsregel; Zahlen von dort stehen hier nicht), `output-analysis-demo` (Warm-up-Erkennung), `sequential-test-demo` (Zwischenblicke), `forecast-interval-demo`, `robust-filter-demo` (derselbe Schutz in anderer Gestalt: hier die Mindestlänge), `wavelet-schrumpfung-demo` (die universale Schwelle hat dieselbe Form σ√(2 ln n) wie √β bei c = 2; die Gleichheit der Konstruktionen ist nicht geprüft und wird nicht behauptet).
- **Nutzen in Betrieb:** keine Aussage über Logistik, Lager oder Fertigung.
- **PDF-Export:** nicht gebaut (keine Entscheidungsvorlage; die Seite ist ein Messbericht).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
python tools/sgo_make_fenster.py --check     # gefrorene Tabellen gegen eine kleine Neurechnung (ebenso sgo_make_online.py, sgo_make_verfahren.py)
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
