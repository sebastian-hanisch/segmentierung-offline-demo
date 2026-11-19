"""Texte der App. Zahlen, die aus Messungen stammen, werden aus den gefrorenen Tabellen gelesen (`sgo_reference`), nicht von Hand getippt; so stehen Text und Messung nicht auseinander."""

import sgo_constants as C
import sgo_reference as REF
from sgo_evaluation import de, pct

INTRO = """
Eine Reihe von Messwerten ist **im Rückblick** gegeben, ein Mittelwertsprung (ein **Bruch**, auch Änderungspunkt) kann irgendwo darin liegen. Gesucht ist die **beste Zerlegung**
in Strecken mit gleichem Mittel. Das ist **exakt berechenbar** (Dynamische Programmierung), und trotzdem hängt das Ergebnis an einer Zahl, die die Mathematik nicht liefert: der **Strafe je Bruch**.
Ohne Strafe wäre jeder Messwert ein eigener Bruch. Die Seite zeigt in fünf Experimenten, wo die Brüche liegen (1), warum die Zahl der Brüche eine **Wahl** und kein Messergebnis ist (2 und 3),
ob schnelle Verfahren dasselbe finden wie das Optimum (4) und was der Rückblick sagt, was ein Alarm in der Zeit nicht sagt (5).

Die Reihen sind **ausgedachte** Normalreihen mit bekannter Streuung 1 (Vehikel A der Reihe „Änderungserkennung“, keine Betriebsdaten). Gezählt werden **Läufe** (simulierte Reihen), **Zerlegungen** und
**Kostenauswertungen**; eine ARL gibt es für die Zerlegung nicht. Weiter unten: **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""

SO_FUNKTIONIERT = """
1. **Kosten einer Strecke:** Summe der Abweichungsquadrate um ihr Mittel. Eine Zerlegung in Strecken hat die Summe dieser Kosten plus **β je Bruch** (die Strafe); das ist ihr **Zielwert**.
2. **Optimale Zerlegung (Dynamische Programmierung):** F(t) = kleinster Zielwert der ersten t Beobachtungen; man probiert für jeden Startpunkt s der letzten Strecke F(s) + Kosten(s, t) + β. Das ist exakt, ohne Abkürzung.
3. **Die Strafe** setzt hier β = c · ln n (c = 1: ein Parameter je Bruch; **c = 2 heißt „BIC-Wahl“: eine Setzung dieses Berichts, keine Quellenaussage**) oder absolut. Zu klein: Fehlbrüche; zu groß: wahre Brüche fallen weg.
4. **Schnelle Verfahren:** PELT streicht Startpunkte, die nie mehr optimal werden können; die **binäre Segmentierung** teilt erst die ganze Reihe an der besten Stelle und dann jede Hälfte, solange ein Teilen mehr als β spart.
   Beide können vom Optimum abweichen (Experiment 4).
5. **Mindestlänge** m je Strecke ist eine zweite Stellschraube: sie schützt vor Einzelausreißern und kostet kurze Brüche.
"""


def exp1_frage() -> str:
    return ("**Frage 1: Wo liegen die Brüche, und wie viele sind es?** Wählen Sie ein Szenario (Sprung, Treppe, Puls, Zickzack, Nullreihe) oder tragen Sie eine eigene Reihe ein, dann Strafe, Mindestlänge "
            "und Verfahren. Das Verdikt vergleicht den gefundenen Zustand mit dem wahren; bei n ≤ 14 rechnet die Aufzählung aller Zerlegungen mit und entscheidet zweitens.")


def exp1_negativ() -> str:
    return ("**Negativer Befund:** Ein einzelner Lauf sagt nichts über die Güte des Verfahrens: dieselbe Strafe findet in einer Reihe alles richtig und in der nächsten einen Bruch zu viel. "
            "Das zeigen Experiment 2 und 3 über viele Läufe. Der Zielwert des gewählten Verfahrens steht in der Tabelle neben den drei anderen; die optimale Zerlegung hat nie einen größeren.")


def exp2_frage() -> str:
    return ("**Frage 2: Ist die Zahl der Brüche ein Messergebnis oder eine Strafwahl?** Der Anteil der Läufe mit richtiger Bruchzahl gegen die Strafkonstante c: unten gefroren aus drei Zufalls-Seeds "
            "mit je 1000 Läufen, oben live mit dem R der Seitenleiste. Ein **Fenster** ist der Bereich von c, in dem mindestens 90 % der Läufe die richtige Zahl finden.")


def exp2_falle() -> str:
    nl = REF.fenster_pooled("null", 0.0)
    zz = REF.fenster_pooled("zickzack", 1.0)
    ein = REF.fenster_pooled("ein", 1.0)
    cg = nl["c_grid"]
    i1, i2 = cg.index(1.0), cg.index(2.0)
    return (f"**Falle: die Strafe ist nicht aus den Daten abzulesen.** Nullreihe, n = 200, Mindestlänge 5 (gefroren, drei Zufalls-Seeds je 1000 Läufe): bei c = 1 haben "
            f"{de(100 * nl['p_any']['lo'][i1], 1)} bis {de(100 * nl['p_any']['hi'][i1], 1)} % der Läufe mindestens einen Fehlbruch (mittlere Zahl {de(nl['mean_count']['mean'][i1], 2)}), bei c = 2 "
            f"{de(100 * nl['p_any']['lo'][i2], 1)} bis {de(100 * nl['p_any']['hi'][i2], 1)} %. Ein Bruch mit δ = 1: Fenster c = {min(ein['window']):g} bis {max(ein['window']):g}. "
            f"Zickzack mit δ = 1: **kein c erreicht 90 %** (hier mit n = 200, Mindestlänge 5, bekannter Streuung; bei c = 2 {de(100 * zz['p_right']['lo'][i2], 1)} bis {de(100 * zz['p_right']['hi'][i2], 1)} %, bei größerem c fallen wahre Brüche weg: "
            f"c = 4 nur {de(100 * zz['p_right']['mean'][cg.index(4.0)], 1)} %). Die obere Kante hängt an Sprunghöhe und Zahl der Brüche, die untere ist scharf; die Mindestlänge ist eine zweite "
            f"Stellschraube (Tabelle unten).")


def exp2_mindestlaenge_text() -> str:
    z = REF.fenster()["mindestlaenge"]
    return (f"**Mindestlänge als zweite Stellschraube** (Nullreihen, n = 200, R = 1000, Zufalls-Seed {REF.fenster()['meta']['mindestlaenge']['seed']}): Anteil der Läufe mit mindestens einem Fehlbruch bei c = 1 / 2 / 3: "
            + "; ".join(f"Mindestlänge {m}: {pct(v[0])} / {pct(v[1])} / {pct(v[2])}" for m, v in z.items())
            + ". Sie schützt vor Einzelausreißern, kostet aber kurze Brüche: ein Puls der Höhe 3 und c = 2, Mindestlänge 5, wird mit der Breite "
            + ", ".join(f"{w}: {pct(v['p_right'])}" for w, v in REF.fenster()["puls_breite"].items()) + " richtig gezählt.")


def exp3_frage() -> str:
    return ("**Frage 2, exakt gerechnet:** Wie wahrscheinlich ist ein Fehlbruch in einer Reihe ohne Bruch? Für n = 2 und n = 3 gibt es eine exakte Antwort (zwei unabhängige Wege), für n ≥ 4 nicht: "
            "dort bleibt die Simulation. Der Gewinn einer Teilung an einer **festen** Stelle ist χ² mit einem Freiheitsgrad; die DP prüft aber alle Stellen zugleich.")


def exp3_negativ() -> str:
    r = REF.null_n()["n_reihe"]
    n200 = r["200"]
    return (f"**Keine Schranke:** „Zahl der Stellen × Einzelwert“ ist für n = 200 bei c = 1 gleich {de(100 * n200['1.0']['stellen_mal_einzel'], 0)} % und bei c = 2 {de(100 * n200['2.0']['stellen_mal_einzel'], 1)} %: "
            f"als Wahrscheinlichkeit wertlos, weil die Gewinne benachbarter Stellen stark korreliert sind. Gemessen sind es {pct(n200['1.0']['p_fehlbruch'])} und {pct(n200['2.0']['p_fehlbruch'])} der Läufe "
            f"(R = 1000). Mit wachsendem n **steigt** der Anteil bei c = 1 ({pct(r['50']['1.0']['p_fehlbruch'])} bei n = 50 bis {pct(r['800']['1.0']['p_fehlbruch'])} bei n = 800) und **fällt** bei c = 2 "
            f"({pct(r['50']['2.0']['p_fehlbruch'])} bis {pct(r['800']['2.0']['p_fehlbruch'])}; belastbar ist der Trend, die Einzelwerte haben bei R = 1000 und einem Zufalls-Seed etwa 1,5 Prozentpunkte Standardfehler): die Strafe c · ln n wächst langsamer als die Zahl der Stellen.")


def exp4_frage() -> str:
    return ("**Frage 3: Findet das schnelle Verfahren dasselbe wie das optimale?** PELT streicht Startpunkte mit einer Regel, die bei Mindestlänge 1 beweisbar nichts Optimales streicht (Killick u. a.); "
            "die binäre Segmentierung teilt gierig. Gezählt werden Läufe, in denen der Zielwert größer ist als der der optimalen Zerlegung, und Kostenauswertungen.")


def exp4_falle() -> str:
    v = REF.verfahren()
    n5 = v["n5_exakt"]
    z = v["zufall_exakt"]
    p = v["pelt_n200"]
    b = v["binseg_n200"]["gesamt"]
    return (f"**Falle: „PELT ist exakt und linear“ gilt nur unter Bedingungen.** Exakt gerechnet (alle 1024 ganzzahligen Reihen mit n = 5 und Werten 0 bis 3, Strafe 1 und 2, zusammen 2048 Instanzen): "
            f"Mindestlänge 1: PELT weicht in {n5['1']['pelt_abw_beta1'] + n5['1']['pelt_abw_beta2']} Instanzen ab; Mindestlänge 2: in {n5['2']['pelt_abw_beta1']} (β = 1) und {n5['2']['pelt_abw_beta2']} (β = 2) von 1024. "
            f"Bei zufälligen ganzzahligen Reihen mit n ≤ 16 (je {z['1']['instanzen']} Instanzen, Zufallsquelle der Demo) weicht PELT bei Mindestlänge 2 / 3 / 5 in {z['2']['pelt_abw']} / {z['3']['pelt_abw']} / {z['5']['pelt_abw']} ab; "
            f"bei n = 200 und Mindestlänge 5 in {p['5']['pelt_abw']} von {p['5']['instanzen']} Instanzen (acht Zufalls-Seeds), bei Mindestlänge 2 in {p['2']['pelt_abw']} von {p['2']['instanzen']}. "
            f"Das **verzögerte Streichen** (eigener Vorschlag, bewiesen für den Zielwert, siehe 📐) weicht in diesen Instanzen nie ab. Die binäre Segmentierung liefert bei n = 200 (13 Szenario-Zellen, c = 1, 2, 4) "
            f"in {de(100 * b['anteil_andere'], 1)} % der Läufe eine andere Bruchmenge, nie eine bessere; **das Mittel hängt an dieser gesetzten Mischung** (je Zelle 0 bis über 70 %).")


def exp5_frage() -> str:
    return ("**Frage 4: Was sagt der Rückblick, was der Alarm nicht sagt, und umgekehrt?** Dieselbe Reihe, zwei Fragen: der einseitige CUSUM (k = 0,5, Schwelle h für ARL₀ = 370,398, zero-state) "
            "meldet in der Zeit; die optimale Zerlegung (c = 2, Mindestlänge 5) schaut auf die fertige Reihe. **Jede Zahl steht mit ihrem Maß.**")


def exp5_negativ() -> str:
    s = REF.online()["simulation"]["1.0"]
    s2 = REF.online()["simulation"]["2.0"]
    return (f"**Bei δ = 1 kein Sieg einer Seite:** Online zahlt man Verzögerung und Fehlalarme (hier {pct(REF.online()['exakt']['verzoegerung']['1.0']['p_alarm_vor_nu'], 2)} Alarm vor ν = 100), offline die Strafwahl und die ganze Reihe: "
            f"offline sind es alle {C.N5} Beobachtungen, durch die Reihenlänge gesetzt, kein Messwert. Gleiche Läufe (kein Alarm vor ν, {s['gleiche_laeufe_n']} von {s['R']}, Zufalls-Seed {s['seed']}): "
            f"mittlerer Lagefehler online {de(s['gleiche_online_lage_betrag'], 2)}, offline bei genau einem Bruch {de(s['gleiche_offline_lage_betrag'], 2)} (δ = 1). Das ist bei δ = 1 ein Nebeneinander "
            f"zweier Maße. Bei δ = 2 ist der Rückblick bei der Lage genauer (gleiche Läufe: online {de(s2['gleiche_online_lage_betrag'], 2)}, offline {de(s2['gleiche_offline_lage_betrag'], 2)}).")


ASSUMPTIONS_MARKDOWN = """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Mittelwertsprünge**, Streuung **bekannt (1)**, unabhängiges Normalrauschen | Unbekannte Streuung: Schalter in Experiment 2 (robuste Schätzung aus den Differenzen); Autokorrelation und Schiefe (Romano u. a. 2022: Standardstrafen überschätzen die Bruchzahl bei autokorreliertem Rauschen) sind **nicht gemessen** | Stück 5 (abhängige Werte), Stück 8 (Lastwechsel am Gate) |
| **Ausgedachte Reihen** (Vehikel A) | In den gelesenen Quellen gibt es keine reale Datenquelle für diese Reihe; was ein Rückblick-Verfahren oder eine Alarmregel in Logistik, Lager oder Fertigung nützt, wird hier **nicht behauptet** | – |
| **Eine Reihe, ein Kanal**, Mittelwert springt | Varianzwechsel, mehrere Kanäle, nichtparametrische Kosten, Wild Binary Segmentation und BOCPD sind **nicht gemessen** | offen (Stück 6 für viele Kanäle) |
| **Strafe β = c · ln n**, Mindestlänge m | Die Zahl der Brüche folgt aus dieser Wahl; c = 2 („BIC-Wahl“) ist eine Setzung dieses Berichts | Experiment 2 und 3 |
| **PELT mit K = 0** | Die Aussage „PELT weicht bei Mindestlänge über 1 ab“ gilt **für diese Implementierung der Beschneidungsregel**; ob der Satz von Killick u. a. die Mindestlänge ausschließt, ist nicht geprüft | offen |
| **Online-Seite: einseitiger CUSUM** | Der Alarm geht nur nach oben, der Rückblick unterscheidet die Richtung nicht; Lorden- und Shiryaev-Verzögerung sind hier nicht gemessen | Stück 2 (CUSUM gegen Karte) |
| **Phase I als Offline-Suche** (Tab in Experiment 5) | Die Folge für die Karte (bedingte ARL₀ nach einer Phase I mit verdecktem Sprung) ist **nicht gemessen** | Stück 4 (Parameter schätzen) |
| **Rechenumgebung** | Gefrorene Zahlen unter Windows mit Python 3.14 gerechnet; Normalzahlen über plattformabhängige Umkehrfunktion (Tests prüfen Bänder); Rechenzeiten auf Streamlit Cloud nicht gemessen | – |
"""

MATH_INTRO = r"""
**Lesehilfe.** *Gegeben:* eine Reihe $x_1, \dots, x_n$, eine Strafe $\beta \ge 0$ je Bruch und eine Mindestlänge $m \ge 1$ je Strecke. *Gesucht:* die Zerlegung, die den Zielwert minimiert.

| Symbol | Bedeutung |
|---|---|
| $x_1, \dots, x_n$ | Beobachtungen; Modell: stückweise konstanter Mittelwert, Normalrauschen mit Streuung 1 |
| $s$ (Bruch) | Zahl der Beobachtungen **vor** der Änderung: $x_1 \dots x_s$ liegen vor, $x_{s+1} \dots$ nach dem Bruch ($\nu = s$) |
| $C(s,t)$ | Kosten der Strecke $(s, t]$: Summe der Abweichungsquadrate um ihr Mittel |
| $\beta$, $c$ | Strafe je Bruch; hier $\beta = c \ln n$ |
| $m$ | Mindestlänge einer Strecke |
| $F(t)$ | kleinster Zielwert der ersten $t$ Beobachtungen, $F(0) = -\beta$ |
"""

CONVENTIONS_MARKDOWN = """
| Begriff | Konvention dieser Demo |
|---|---|
| **Bruch** | Änderungspunkt ν = Zahl der Beobachtungen vor der Änderung (die letzte Beobachtung vor der Änderung, wie Moustakides und Bartroff; Lorden zählt die erste danach: Unterschied 1) |
| **Strafe** | β = c · ln n; c = 1: ein Parameter je Bruch; c = 2 („BIC-Wahl“) ist eine Setzung dieses Berichts, keine Quellenaussage; andere Kostenskala als Truong u. a. (Faktor 2 zwischen den Konventionen) |
| **Mindestlänge** | Jede Strecke hat mindestens m Beobachtungen; zweite Stellschraube neben β |
| **Fehlbruch gegen Fehlalarm** | Fehlbruch: ein gefundener Bruch ohne wahren Bruch (Läufe gezählt); Fehlalarm: nur ein Alarm der Online-Karte vor ν (je Zeit gezählt, ARL₀) |
| **Zähleinheiten** | Läufe (eine simulierte Reihe), Zerlegungen (2ⁿ⁻¹ bei m = 1, Fibonacci-Zahlen bei m = 2), Kostenauswertungen (zulässige Strecken, die ein Verfahren ausrechnet), Beobachtungen |
| **Gleichstand** | Der kleinste Startpunkt s der letzten Strecke gewinnt; bei ganzzahligen Reihen mit exaktem Gleichstand kann die Gleitkomma-Rechnung eine andere der gleichwertigen Zerlegungen wählen, der Zielwert ist gleich |
| **Zentrieren** | Jede Reihe wird vor der Rechnung um ihr Mittel verschoben (ändert die Zerlegung mathematisch nicht, verhindert Stellenverlust bei großem Mittel) |
| **ARL₀-Ziel (Experiment 5)** | 370,398 = 1/(2 Φ(−3)), nicht das gesetzte 370; zero-state; Alarm bei C ≥ h |
"""

MATH_MARKDOWN = r"""
**Zielwert einer Zerlegung** mit Brüchen $0 = s_0 < s_1 < \dots < s_k < s_{k+1} = n$ und $s_{j+1} - s_j \ge m$:

$$\sum_{j=0}^{k} C(s_j, s_{j+1}) + \beta\,k, \qquad C(s,t) = \sum_{i=s+1}^{t} \left(x_i - \bar x_{(s,t]}\right)^2 = \sum_{i=s+1}^{t} x_i^2 - \frac{\left(\sum_{i=s+1}^{t} x_i\right)^2}{t-s}.$$

**Optimale Zerlegung (Dynamische Programmierung, Optimal Partitioning):**

$$F(0) = -\beta, \qquad F(t) = \min_{s}\; \bigl[\,F(s) + C(s,t) + \beta\,\bigr] \quad \text{über } s \text{ mit } s = 0 \text{ oder } s \ge m,\; t - s \ge m.$$

Rückverfolgung vom letzten $s$; bei Gleichstand gewinnt das kleinste $s$. Der Aufwand ist quadratisch in $n$.

**Gewinn einer Teilung** der Strecke $(a, b]$ bei $\tau$ (Gewinn = Kosten vorher minus Kosten nachher):

$$C(a,b) - C(a,\tau) - C(\tau,b) = \frac{(\tau - a)(b - \tau)}{b - a}\left(\bar x_{(a,\tau]} - \bar x_{(\tau,b]}\right)^2 .$$

Die **binäre Segmentierung** teilt eine Strecke bei dem $\tau$ mit dem größten Gewinn, **nur wenn** dieser Gewinn größer als $\beta$ ist, und verfährt mit beiden Hälften ebenso. Sie sieht nur einen Bruch auf einmal:
beim Puls $(0,0,0,2,2,2,0,0,0)$ ist der größte Einzelgewinn 2, beide Brüche zusammen entfernen aber alle Kosten 8; für $2 \le \beta < 4$ findet die optimale Zerlegung beide Brüche, die binäre Segmentierung keinen.

**PELT** (Killick, Fearnhead, Eckley 2012) streicht den Startpunkt $s$ bei $t$, wenn

$$F(s) + C(s,t) \;\ge\; F(t) \qquad (K = 0).$$

Bei Mindestlänge 1 ist das beweisbar ohne Verlust (Satz 3.1 der Quelle, nach Abstract und Abschnitten 1 bis 4 gelesen). Bei $m \ge 2$ ist der Ersatzweg „über $t$“ nicht immer zulässig, weil die letzte Strecke $(t, T]$ kürzer als $m$ sein kann:
im Beispiel $x = (2,1,0,0,2)$, $m = 2$, $\beta = 1$ streicht die Regel bei $t = 4$ den Start 0, und PELT endet bei Zielwert $25/6$ mit einem Bruch statt bei 4 ohne Bruch.

**Verzögertes Streichen** (eigener Vorschlag dieses Stücks, **keine Quellenaussage**): ein markierter Startpunkt $s$ wird erst bei $T \ge t + m$ gestrichen. *Beweis für den Zielwert:* gilt
$F(s) + C(s,t) \ge F(t)$ und $C(s,t) + C(t,T) \le C(s,T)$ (Quadratsummenkosten), so folgt $F(s) + C(s,T) \ge F(t) + C(t,T)$; der Weg über $t$ ist für jedes $T \ge t + m$ zulässig,
und die letzte Strecke $(t, T]$ hat dann mindestens die Länge $m$. Ist $t$ selbst vorher gestrichen worden, ersetzt man es nach demselben Schluss durch dessen Ersatz (Induktion über die Streichzeit).
Bei Gleichständen kann die Bruchmenge abweichen, der Zielwert nicht.

**Nullreihe, $n = 2$** (Mindestlänge 1): ein Fehlbruch genau dann, wenn $(x_1 - x_2)^2/2 > \beta$, also mit $\chi^2_1$:

$$P(\text{Fehlbruch}) = P(\chi^2_1 > \beta) = 2\,\bar\Phi\!\left(\sqrt{\beta}\right) = \operatorname{erfc}\!\left(\sqrt{\beta/2}\right).$$

**Nullreihe, $n = 3$:** kein Bruch genau dann, wenn beide Teilungsgewinne höchstens $\beta$ sind und die Gesamtkosten höchstens $2\beta$ (die Gewinne sind mit $\rho = 1/2$ korreliert, die Teilungsrichtungen schneiden sich unter $60^\circ$):

$$P(\text{kein Bruch}) = \frac{1}{\pi}\int_0^{\pi} \left[1 - \exp\!\left(-\tfrac12 \min\!\left\{2\beta,\; \frac{\beta}{\cos^2\varphi},\; \frac{\beta}{\cos^2(\varphi - 60^\circ)}\right\}\right)\right] d\varphi .$$

**F1 mit Toleranz $M$:** ein gefundener Bruch gilt als getroffen, wenn ein wahrer Bruch **näher als $M$** liegt; Genauigkeit = getroffene gefundene Brüche durch gefundene, Ausbeute = getroffene wahre Brüche durch wahre,
$F_1 = 2PR/(P+R)$; $F_1 = 0$, wenn nichts getroffen oder nichts gefunden wird; ohne wahren Bruch nicht definiert (Festlegung dieses Berichts). $M$ sollte kleiner als der kleinste Abstand wahrer Brüche sein (Truong u. a.); hier verschärft auf den halben Abstand
(eigene Verschärfung, keine Quellenaussage).

**Online-Seite (Experiment 5):** $C_t = \max(0,\, C_{t-1} + x_t - k)$, $k = 0{,}5$, Alarm beim ersten $t$ mit $C_t \ge h$; $h$ so, dass die zero-state-ARL₀ gleich $370{,}398$ ist (Integralgleichung, geprüft gegen die Brook/Evans-Kette).

**Literatur** (Stufe **W** = gelesen, sonst „nicht gelesen“): Killick, Fearnhead, Eckley 2012, arXiv 1101.1438 (**W**, Abstract und Abschnitte 1 bis 4); Truong, Oudre, Vayatis 2020 (**W**, Abschnitte 2, 3.2, 5.1); Van den Burg, Williams (**W**);
Romano u. a. 2022 (**W**, Abstract und Einleitung); Bai, Perron (**W**, nur die Arbeitsfassung CIREQ 9807). **Nicht gelesen:** Basseville/Nikiforov, Page 1954, Jackson u. a. 2005, Scott/Knott 1974 im Original.
"""
