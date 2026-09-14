#include "CPPBlackBoard.h"

CPPBlackBoard::CPPBlackBoard()
{
	RunningTime = 0;
	DeltaSecond = 0.0166666;

	MyLocation_Cartesian		= Vector3(0,0,0);
	TargetLocaion_Cartesian		= Vector3(0, 0, 0);
	VP_Cartesian				= Vector3(0, 0, 0);

	MyForwardVector = Vector3(0, 0, 0);
	MyUpVector		= Vector3(0, 0, 0);
	MyRightVector	= Vector3(0, 0, 0);

	TargetForwardVector = Vector3(0, 0, 0);
	TargetUpVector		= Vector3(0, 0, 0);
	TargetRightVector	= Vector3(0, 0, 0);

	MyRotation_EDegree		= EulerAngle(0,0,0);
	TargetRotation_EDegree	= EulerAngle(0, 0, 0);

	MySpeed_MS		= 0;
	TargetSpeed_MS	= 0;
	MyBodyVelocity = Vector3(0, 0, 0);
	TargetBodyVelocity = Vector3(0, 0, 0);
	MyAOA_Degree = 0;
	MyAOS_Degree = 0;
	MyNz = 1;
	MyKCAS_KT = 0;
	TargetKCAS_KT = 0;
	HasPerfectMyState = false;
	HasPerfectTargetState = false;
	MyHealth = 1;
	TargetHealth = 1;

	Distance = 0;
	Throttle = 1;
	TargetSpeedCommand_MS = 220;
	ThrottleCommandMode = THR_MAX;


	Los_Degree = 0;
	Los_Degree_Target = 0;

	MyAngleOff_Degree = 0;
	MyAspectAngle_Degree = 0;

	BFM = NONE;
	ACM = EF;

	Team = UNKNOWN;

	AltSpeed = 0;
	ClosureRate_MS = 0;
	TimeToMerge = 999;
	EnergyAdvantage_M = 0;
	DamageDifference = 0;
	EstimatedDamageDealt = 0;
	EstimatedDamageTaken = 0;
	MyDamageRate = 0;
	EnemyDamageRate = 0;
	TargetInMyCone = false;
	OwnshipInEnemyCone = false;
	NeutralTime = 0;
	ThreatClearTime = 0;
	OffensiveCommitUntil = 0;
	DefensiveCommitUntil = 0;
	DefensiveRecoveryUntil = 0;
	ShotCommitUntil = 0;
	ManeuverCooldownUntil = 0;
	HABFMPullToHUDUntil = 0;
	HABFMNextManeuverTask = -1;
	RejoinCommitUntil = 0;
	RejoinTurnSide = 0;
	RejoinTaskKind = -1;
	DefensiveTurnCommitUntil = 0;
	DefensiveTurnSide = 0;
	Phase = 1;
	EnemyPursuitType = 0;
	ControlZoneState = 0;
	ControlZoneDwell = 0;
	Node35Until = 0;
	Node35State = 0;
	TrackSubMode = 0;
	VPPMode = 0;
	MyDamageBand = 0;
	EnemyDamageBand = 0;
	TargetTurnRadius_M = 1000;
	TargetTurnRate_DegSec = 0;
	TargetTurnCircleValid = false;

	MyLosRate_DegSec = 0;
	PreviousMyLos_Degree = 0;
	EnemyLosRate_DegSec = 0;
	PreviousEnemyLos_Degree = 0;
	VPVelocity = Vector3(0, 0, 0);
	VPVelocityValid = false;
	VppBlendWeight = 0.5f;
	PreviousPNAnchor = Vector3(0, 0, 0);
	PreviousPNTaskKind = -1;
	PreviousVppTaskKind = -1;
	HasPreviousPNAnchor = false;
	VelocityPointingController = false;
	LastControllerModeUsed = 0;
	TerminalNoseAimActive = false;
	PaperTerminalAccelerationActive = false;
	AlphaBiasApplied = false;
	AlphaBiasAngleDeg = 0;
	AlphaBiasWeight = 0;
	TrackSubReason = 0;
	ThrottleReason = 0;
	SignedVPErrorDeg = 0;
	SignedTargetErrorDeg = 0;
	NoseToTargetSignedDeg = 0;
	NoseToVPSignedDeg = 0;
	TargetErrorHorizontalDeg = 0;
	TargetErrorVerticalDeg = 0;
	VPErrorHorizontalDeg = 0;
	VPErrorVerticalDeg = 0;
	RollSaturated = false;
	PitchSaturated = false;
	RudderSaturated = false;
	ActiveCompetitionTask = -1;
	LockedManeuverTask = -1;
	ManeuverTurnDegrees = 0;
	ManeuverStartTime = 0;
	LockedManeuverSide = 0;
	PendingScissors = false;
	PreviousManeuverForward = Vector3(0, 0, 0);
	PreviousLevelForward = Vector3(0, 0, 0);
	PreviousTargetLevelForward = Vector3(0, 0, 0);
	PlanarTurnRate_DegSec = 0;
	PlanarTurnRateSigned_DegSec = 0;
	TargetPlanarTurnRateSigned_DegSec = 0;
	CircleDirectionRelation = 0;
	LastManeuverCueOutcome = CUE_NONE;
	EvaluatedConditionMask = 0;
	TrueConditionMask = 0;
	LastRollCommand = 0;
	LastPitchCommand = 0;
	LastRudderCommand = 0;
	IsAimmingMode = false;
}

CPPBlackBoard::~CPPBlackBoard()
{
}
