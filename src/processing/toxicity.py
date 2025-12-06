"""
Text toxicity/content moderation using OpenAI Moderation API.

The Moderation API checks for:
- hate: Hateful content
- hate/threatening: Hateful + threatening
- harassment: Harassing content  
- harassment/threatening: Harassing + threatening
- self-harm: Self-harm content
- self-harm/intent: Intent to self-harm
- self-harm/instructions: Instructions for self-harm
- sexual: Sexual content
- sexual/minors: Sexual content involving minors
- violence: Violent content
- violence/graphic: Graphic violence
"""
import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class ModerationScore:
    """Moderation scores from OpenAI API."""
    hate: float
    hate_threatening: float
    harassment: float
    harassment_threatening: float
    self_harm: float
    self_harm_intent: float
    self_harm_instructions: float
    sexual: float
    sexual_minors: float
    violence: float
    violence_graphic: float
    
    # Overall flags
    flagged: bool
    
    @property
    def max_score(self) -> float:
        """Return the highest score across all categories."""
        return max(
            self.hate, self.hate_threatening,
            self.harassment, self.harassment_threatening,
            self.self_harm, self.self_harm_intent, self.self_harm_instructions,
            self.sexual, self.sexual_minors,
            self.violence, self.violence_graphic
        )
    
    @property
    def severity(self) -> str:
        """Get severity level based on max score."""
        score = self.max_score
        if score < 0.3:
            return "safe"
        elif score < 0.5:
            return "mild"
        elif score < 0.7:
            return "moderate"
        elif score < 0.9:
            return "severe"
        else:
            return "extreme"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "hate": round(self.hate, 4),
            "hate_threatening": round(self.hate_threatening, 4),
            "harassment": round(self.harassment, 4),
            "harassment_threatening": round(self.harassment_threatening, 4),
            "self_harm": round(self.self_harm, 4),
            "self_harm_intent": round(self.self_harm_intent, 4),
            "self_harm_instructions": round(self.self_harm_instructions, 4),
            "sexual": round(self.sexual, 4),
            "sexual_minors": round(self.sexual_minors, 4),
            "violence": round(self.violence, 4),
            "violence_graphic": round(self.violence_graphic, 4),
            "max_score": round(self.max_score, 4),
            "flagged": self.flagged,
            "severity": self.severity
        }


@dataclass
class SegmentModeration:
    """Moderation analysis for a timestamped segment."""
    start: float
    end: float
    text: str
    scores: ModerationScore


def get_openai_client() -> OpenAI:
    """Get OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


def analyze_toxicity(text: str) -> ModerationScore:
    """
    Analyze text for harmful content using OpenAI Moderation API.
    
    Args:
        text: Text to analyze
    
    Returns:
        ModerationScore with all category scores
    """
    if not text or not text.strip():
        return ModerationScore(
            hate=0.0, hate_threatening=0.0,
            harassment=0.0, harassment_threatening=0.0,
            self_harm=0.0, self_harm_intent=0.0, self_harm_instructions=0.0,
            sexual=0.0, sexual_minors=0.0,
            violence=0.0, violence_graphic=0.0,
            flagged=False
        )
    
    client = get_openai_client()
    response = client.moderations.create(input=text)
    
    result = response.results[0]
    scores = result.category_scores
    
    return ModerationScore(
        hate=scores.hate,
        hate_threatening=getattr(scores, 'hate/threatening', 0.0),
        harassment=scores.harassment,
        harassment_threatening=getattr(scores, 'harassment/threatening', 0.0),
        self_harm=getattr(scores, 'self-harm', 0.0),
        self_harm_intent=getattr(scores, 'self-harm/intent', 0.0),
        self_harm_instructions=getattr(scores, 'self-harm/instructions', 0.0),
        sexual=scores.sexual,
        sexual_minors=getattr(scores, 'sexual/minors', 0.0),
        violence=scores.violence,
        violence_graphic=getattr(scores, 'violence/graphic', 0.0),
        flagged=result.flagged
    )


def analyze_segments(segments: list[dict]) -> list[SegmentModeration]:
    """
    Analyze moderation for multiple timestamped segments.
    
    Args:
        segments: List of dicts with 'start', 'end', 'text' keys
    
    Returns:
        List of SegmentModeration results
    """
    results = []
    
    for seg in segments:
        text = seg.get("text", "").strip()
        scores = analyze_toxicity(text)
        
        results.append(SegmentModeration(
            start=seg["start"],
            end=seg["end"],
            text=text,
            scores=scores
        ))
    
    return results


def get_flagged_segments(
    segments: list[SegmentModeration],
    threshold: float = 0.5
) -> list[SegmentModeration]:
    """
    Filter segments to only those exceeding threshold.
    
    Args:
        segments: List of analyzed segments
        threshold: Score threshold (0.0 - 1.0)
    
    Returns:
        List of segments with max_score >= threshold
    """
    return [seg for seg in segments if seg.scores.max_score >= threshold]


def aggregate_moderation(segments: list[SegmentModeration]) -> dict:
    """
    Aggregate moderation scores across all segments.
    """
    if not segments:
        return {
            "max_score": 0.0,
            "avg_score": 0.0,
            "flagged_count": 0,
            "total_segments": 0,
            "flagged_percentage": 0.0,
            "overall_severity": "safe"
        }
    
    max_scores = [seg.scores.max_score for seg in segments]
    flagged_count = sum(1 for seg in segments if seg.scores.flagged)
    
    max_score = max(max_scores)
    avg_score = sum(max_scores) / len(max_scores)
    
    # Determine severity
    if max_score < 0.3:
        severity = "safe"
    elif max_score < 0.5:
        severity = "mild"
    elif max_score < 0.7:
        severity = "moderate"
    elif max_score < 0.9:
        severity = "severe"
    else:
        severity = "extreme"
    
    return {
        "max_score": round(max_score, 4),
        "avg_score": round(avg_score, 4),
        "flagged_count": flagged_count,
        "total_segments": len(segments),
        "flagged_percentage": round(flagged_count / len(segments) * 100, 2),
        "overall_severity": severity
    }
