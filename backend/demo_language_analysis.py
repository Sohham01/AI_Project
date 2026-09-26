"""
Demo script: runs the full language analysis pipeline on sample texts
and prints detailed results.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from app.services.nlp.language import SimpleLanguageDetector
from app.services.nlp.script_detector import SimpleScriptDetector
from app.services.nlp.code_mix_detector import SimpleCodeMixDetector
from app.services.nlp.language_analysis_service import LanguageAnalysisService


SAMPLES = [
    "Is IIM Sambalpur worth it?",
    "IIM jaana worth it hai kya bhai?",
    "क्या IIM Sambalpur अच्छा है?",
    "आईआईएम संबलपुर अच्छा है",
    "আইআইএম সম্বলপুর কেমন?",
    "ଆଇଆଇଏମ ସମ୍ବଲପୁର ଭଲ କି?",
    "broooo 😭 IIM ka placement kya scene hai??",
    "Bhai placement bahut accha hai IIM Sambalpur ka",
    "আমি IIM Sambalpur তে পড়ি",
    "ମୁଁ IIM Sambalpur ରେ ପଢ଼ୁଛି",
    "lol same",
    "😂😂😂",
    "!!!???",
    "",
]


def main() -> None:
    service = LanguageAnalysisService(
        language_detector=SimpleLanguageDetector(),
        script_detector=SimpleScriptDetector(),
        code_mix_detector=SimpleCodeMixDetector(),
    )

    for text in SAMPLES:
        result = service.analyze(text)
        print("=" * 60)
        print(f"TEXT:                {text!r}")
        print(f"LANGUAGE:            {result.language}")
        print(f"LANGUAGE CONFIDENCE: {result.language_confidence}")
        print(f"SCRIPT:              {result.script}")
        print(f"SCRIPT CONFIDENCE:   {result.script_confidence}")
        print(f"CODE-MIXED:          {result.is_code_mixed}")
        print(f"DETECTED LANGUAGES:  {result.detected_languages}")
        print(f"DETECTOR:            {result.detector_name}")
        print(f"VERSION:             {result.detector_version}")
        print()


if __name__ == "__main__":
    main()
