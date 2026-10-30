# Implementarea hardware

> Corespunde capitolului 3 din teză. Secțiunile 1–3 descriu proiectul din lucrare. Secțiunile 4–7 descriu cablajul validat pe banc după predarea tezei: pini, conexiuni, calibrare și probleme deschise.

## 1. Controlerul central — Raspberry Pi 5

| Parametru | Valoare |
|---|---|
| Procesor | Broadcom BCM2712, 4× Cortex-A76 la 2,4 GHz |
| Cip I/O | RP1: GPIO, SPI, I2C, UART, 4 canale PWM hardware |
| Stocare | microSD (suport NVMe prin PCIe 2.0) |
| Consum | ~5 W în repaus, 8–12 W sub sarcină |

Pi 5 nu are intrări analogice. Senzorii analogici (pH, EC) se citesc printr-un convertor **ADS1115** (16 biți, I2C).

## 2. Senzori

| Parametru | Senzor | Interfață | Specificații principale |
|---|---|---|---|
| pH | sondă Ag/AgCl + modul pH-4502C | analogic → ADS1115 A0 | 0–14 pH, ±0,1 pH, calibrare în 3 puncte |
| EC | modul EC/TDS cu doi electrozi | analogic → ADS1115 A1 | 0–10 mS/cm, ±2%, etalon 1413 µS/cm |
| Temperatură apă | DS18B20 waterproof | 1-Wire (GPIO4) | −55…+125 °C, ±0,5 °C, 12 biți |
| Temperatură și umiditate aer | DHT22 (AM2302) | single-wire digital | ±0,5 °C, ±2–5% RH, 0,5 Hz |
| Luminozitate | BH1750FVI | I2C (0x23) | 1–65.535 lux |
| Nivel bazin | plutitor magnetic + senzor Hall | digital / analogic | fără contact cu lichidul |

## 3. Actuatoare

| Actuator | Rol | Comandă |
|---|---|---|
| 2× pompă peristaltică | dozare nutrient A și nutrient B | punte H dublă (MOSFET), `gpiozero.Motor` |
| Pompă de recirculare (externă, inline) | pelicula NFT continuă | tranzistor, PWM pe GPIO18 |
| Ventilator | microclimat (T > 28 °C sau RH > 70%) | releu on/off (în teză: PWM 4-pin) |

Pompa de recirculare este externă, nu submersibilă, din două motive. Nu încălzește soluția, iar montată sub nivelul bazinului se reamorsează singură după o cădere de curent. Senzorul de nivel stă deasupra punctului de aspirație, ca operatorul să fie avertizat înainte ca pompa să funcționeze în gol.

## 4. Alocarea pinilor (validată pe banc, 2026-06-30)

| Funcție | Pin BCM | Pin fizic | Stare |
|---|---|---|---|
| I2C SDA (ADS1115, BH1750) | GPIO2 | 3 | conectat |
| I2C SCL | GPIO3 | 5 | conectat |
| DS18B20 (1-Wire) | GPIO4 | 7 | prevăzut |
| Pompă peristaltică 1 (1A / 1B) | GPIO5 / GPIO6 | 29 / 31 | validat |
| Pompă peristaltică 2 (2A / 2B) | GPIO13 / GPIO19 | 33 / 35 | validat |
| Releu ventilator | GPIO16 | 36 | validat |
| DHT22 (date) | GPIO17 | 11 | propus |
| Pompă recirculare (PWM) | GPIO18 | 12 | validat |
| Senzor nivel (plutitor) | GPIO23 | 16 | propus |

## 5. Cablaj

### ADS1115 → Raspberry Pi 5

| ADS1115 | Raspberry Pi 5 |
|---|---|
| VDD | pin 2 — 5V |
| GND | pin 6 — GND |
| SCL | pin 5 — GPIO3 |
| SDA | pin 3 — GPIO2 |
| ADDR | GND → adresa `0x48` |
| A0 | `Po` de la pH-4502C |
| A1 | ieșirea analogică a modulului EC |

ADS1115 este alimentat la 5V, ca să acopere ieșirile 0–5V ale modulelor. Magistrala I2C rămâne la 3,3V prin rezistențele pull-up ale Pi-ului. PGA-ul este setat la ±4,096 V.

### pH-4502C (pinii `To Do Po G G V+`)

| Pin | Conexiune |
|---|---|
| `V+` | 5V |
| `G`, `G` | GND |
| `Po` | ADS1115 A0 |
| `Do`, `To` | nefolosiți (temperatura vine de la DS18B20) |
| BNC | electrodul de pH |

### DS18B20

| DS18B20 | Raspberry Pi 5 |
|---|---|
| VDD | pin 1 — 3V3 |
| GND | pin 9 — GND |
| DATA | pin 7 — GPIO4, cu rezistență de 4,7 kΩ între DATA și 3V3 |

### Configurare `/boot/firmware/config.txt`

```
dtparam=i2c_arm=on     # I2C pentru ADS1115 și BH1750
dtoverlay=w1-gpio      # 1-Wire pentru DS18B20 pe GPIO4
```

Verificare: `i2cdetect -y 1` trebuie să afișeze adresa `48`.

## 6. Calibrare

Procedurile din teză sunt implementate în [`src/calibration/`](../src/calibration/):

- **pH** (`calibrate_ph.py`): soluții tampon de pH 4,01 / 6,86 / 9,18 și regresie liniară pe două segmente. Se repetă la 30 de zile sau la schimbarea soluției.
- **EC** (`calibrate_ec.py`): etalon de 1413 µS/cm, factor K = 1413 / EC măsurat, compensare termică de 2%/°C față de 25 °C.

Coeficienții se salvează în `config/*.json`. Aceste fișiere sunt specifice fiecărui exemplar de senzor și sunt excluse din git.

## 7. Reguli de cablaj și probleme cunoscute

1. **Un singur proces citește ADS1115.** Două procese concurente produc coliziuni I2C: un canal citește valoarea altuia. Bucla de control trebuie să fie singurul proprietar al senzorilor.
2. **Masă comună completă.** Minusul surselor de putere (pompe, ventilator) trebuie legat la GND-ul Pi-ului, nu doar masa de semnal. Altfel curentul de comandă nu are cale de întoarcere și motoarele nu comută.
3. **Zgomot PWM pe pH.** Electrodul de pH are impedanță mare, iar citirea oscilează cât timp pompele funcționează. pH-ul se citește cu pompele oprite.
4. **Sonda de pH:** bulbul de sticlă este fragil. Sonda nu trebuie lăsată să se usuce și nu se păstrează în apă demineralizată, ci în KCl sau în soluție tampon.
5. **Probleme deschise:** alimentarea ventilatorului prin contactele NO/COM ale releului, detecția DS18B20 și integrarea senzorilor DHT22, BH1750 și de nivel.
