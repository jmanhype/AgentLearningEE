"""Guardrails for magicbrush domain generated from benchmark."""

from __future__ import annotations

from typing import Dict, Optional

from guardrails.base import NumericGuardrail, constant_guardrail
from guardrails import register_domain


DOMAIN_GUARDRAILS: Dict[str, NumericGuardrail] = {
    "mb-140513-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-140513-turn2": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-319096-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-319096-turn2": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-393224-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-393224-turn2": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-122896-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-415314-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-415314-turn2": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-75065-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-75065-turn2": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-75065-turn3": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-483587-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-483587-turn2": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-393375-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-393375-turn2": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-341070-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-237569-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-237569-turn2": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-237569-turn3": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-253975-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-557105-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-557105-turn2": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-557105-turn3": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-77596-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-7730-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-7730-turn2": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-369969-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-369969-turn2": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-399759-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-399759-turn2": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-57357-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-57357-turn2": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-57357-turn3": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-553827-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-193680-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-358868-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-535902-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-302107-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-302107-turn2": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-158548-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-158548-turn2": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-458821-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-17320-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-17320-turn2": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-17320-turn3": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-491430-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-163114-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-163114-turn2": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
    "mb-493074-turn1": constant_guardrail(
        instructions='Decode images, compute MSE and SSIM. Return "pass" if MSE<=1500 and SSIM>=0.60.',
        value='pass',
        format='string',
    ),
}


def get_guardrail(task_id: str) -> Optional[NumericGuardrail]:
    return DOMAIN_GUARDRAILS.get(task_id)


register_domain('magicbrush', DOMAIN_GUARDRAILS)

