import { ReasoningEffort } from '$lib/enums';
import type { ReasoningEffortLevel } from '$lib/types';

/**
 * Reasoning effort UI labels.
 * Keys match the ReasoningEffort enum values for type-safe lookups.
 */
export const REASONING_EFFORT_LABELS: Record<string, string> = {
	[ReasoningEffort.DEFAULT]: 'Default',
	[ReasoningEffort.HIGH]: 'High',
	[ReasoningEffort.LOW]: 'Low',
	[ReasoningEffort.XHIGH]: 'XHigh',
	[ReasoningEffort.MEDIUM]: 'Medium',
	[ReasoningEffort.OFF]: 'Off'
};

export const REASONING_EFFORT_LEVELS: ReasoningEffortLevel[] = [
	{ label: 'Default', value: ReasoningEffort.DEFAULT },
	{ label: 'Off', value: ReasoningEffort.OFF },
	{ label: 'Low', value: ReasoningEffort.LOW },
	{ label: 'Medium', value: ReasoningEffort.MEDIUM },
	{ hasInfo: true, label: 'XHigh', value: ReasoningEffort.XHIGH }
];

/**
 * Reasoning effort to token budget mapping.
 * Maps the ReasoningEffort enum values to concrete token counts for the server.
 */
export const REASONING_EFFORT_TOKENS: Record<string, number> = {
	[ReasoningEffort.LOW]: 512,
	[ReasoningEffort.XHIGH]: -1, // unlimited
	[ReasoningEffort.MEDIUM]: 2048
};
