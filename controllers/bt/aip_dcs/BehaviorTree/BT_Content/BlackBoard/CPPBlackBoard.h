#pragma once
#include "../../../Geometry/Vector3.h"
#include "../../../Geometry/EulerAngle.h"
#include <cstdint>
#include <vector>

using namespace BT_Geometry;

enum BFM_Mode
{
	OBFM,
	HABFM,
	DBFM,
	DETECTING,
	SCISSORS,
	NONE

};

enum ACM_Mode
{
	EF,
	SF
};

enum TeamColor
{
	BLUE,
	RED,
	UNKNOWN
};

enum S_BFM_Mode
{
	S_OBFM,
	S_HABFM,
	S_DBFM,
	S_Others
};

enum WeaponMode
{
	Gun,
	Missile
};

enum ThrottleMode
{
	THR_AUTO,
	THR_MAX,
	THR_IDLE
};

// Last completed/aborted HABFM cue. Kept as an integer in diagnostics.
enum ManeuverCueOutcome
{
	CUE_NONE = 0,
	CUE_WIN = 1,
	CUE_NEUTRAL = 2,
	CUE_LOSE = 3,
	CUE_RANGE_ABORT = 4,
	CUE_TIMEOUT = 5,
	CUE_STRONG_WIN = 6,
	CUE_STRONG_LOSE = 7
};

/* Aircraft state exchanged with the behavior tree. */
struct PlaneInfo
{
public:

	Vector3			Location;	// Input may be LLA; internal behavior-tree geometry is Cartesian.

	EulerAngle		Rotation;	//Degree
	Vector3			AngleAcceleration;	//PQR
	Vector3			BodyVelocity;		//body x/y/z velocity, m/s (z positive down)
	float			AOA;				//deg
	float			AOS;				//deg
	float			Nz;					//g
	float			KCAS;				//knots calibrated airspeed
	bool			HasExtendedState;

	float			Speed;		//m/s

	int				Team;		// 0 , 1
	float			Resv0;		// aircraft identifier
	float			Resv1;		//HP
	float			Resv2;		// 0: AI, 1: human

	PlaneInfo()
	{
		Location = Vector3(0, 0, 0);
		Rotation = EulerAngle(0, 0, 0);
		Speed = 0;
		BodyVelocity = Vector3(0, 0, 0);
		AOA = 0;
		AOS = 0;
		Nz = 1;
		KCAS = 0;
		HasExtendedState = false;
		Team = 0;
		Resv0 = 0;
		Resv1 = 0;
		Resv2 = 0;
	}
};

struct MissileTarget
{
public:
	int ListIndex;
	int DISID;
};

/* Shared blackboard for BT conditions, tasks, guidance, and diagnostics.
   Attitude angles are degrees unless noted otherwise. */
class CPPBlackBoard
{
public:
	CPPBlackBoard();
	~CPPBlackBoard();

public:
	double RunningTime;										// simulation running time
	double DeltaSecond;										// BT delta time

	std::vector<PlaneInfo> Friendly;						// friendly aircraft
	std::vector<PlaneInfo> Enemy;							// adversary aircraft

	Vector3 MyLocation_Cartesian;							// ownship Cartesian position
	Vector3 TargetLocaion_Cartesian;						// target Cartesian position
	Vector3 VP_Cartesian;									// pursuit point Cartesian position

	Vector3 MyForwardVector;								// ownship forward vector
	Vector3 MyUpVector;										// ownship up vector
	Vector3 MyRightVector;									// ownship right vector

	Vector3 TargetForwardVector;							// target forward vector
	Vector3 TargetUpVector;									// target up vector
	Vector3 TargetRightVector;								// target right vector

	EulerAngle MyRotation_EDegree;							// ownship attitude, degrees
	EulerAngle TargetRotation_EDegree;						// target attitude, degrees

	Vector3 MyAngleAcceleration;

	float MySpeed_MS;										// ownship speed, m/s
	float TargetSpeed_MS;									// target speed, m/s
	Vector3 MyBodyVelocity;
	Vector3 TargetBodyVelocity;
	float MyAOA_Degree;
	float MyAOS_Degree;
	float MyNz;
	float MyKCAS_KT;
	float TargetKCAS_KT;
	bool HasPerfectMyState;
	bool HasPerfectTargetState;
	float MyHealth;
	float TargetHealth;

	float Distance;											// slant range, m
	float Throttle;											//Throttle, 0~1
	float TargetSpeedCommand_MS;
	ThrottleMode ThrottleCommandMode;


	float Los_Degree;										// ownship nose-to-target LOS angle
	float Los_Degree_Target;								// target nose-to-ownship LOS angle

	float MyAngleOff_Degree;								// heading crossing / angle-off proxy
	float MyAspectAngle_Degree;								// target aspect-angle proxy

	bool EnemyInSight;
	bool EnemyInSight_Target;

	BFM_Mode BFM;											// current BFM mode
	ACM_Mode ACM;											// current ACM mode


	TeamColor Team;											// team color


	float AltSpeed;											// vertical speed
	float ClosureRate_MS;
	float TimeToMerge;
	float EnergyAdvantage_M;
	float DamageDifference;
	float EstimatedDamageDealt;
	float EstimatedDamageTaken;
	float MyDamageRate;									// current phase-aware cone damage rate estimate
	float EnemyDamageRate;
	bool TargetInMyCone;
	bool OwnshipInEnemyCone;
	float NeutralTime;
	float ThreatClearTime;
	float OffensiveCommitUntil;
	float DefensiveCommitUntil;
	float DefensiveRecoveryUntil;                   // brief DBFM exit hold after threat-cleared
	float ShotCommitUntil;                                // brief terminal-aim hold
	float ManeuverCooldownUntil;                          // prevents immediate circle re-lock after a cue/abort
	float HABFMPullToHUDUntil;                            // paper block 18 bridge duration
	int HABFMNextManeuverTask;                            // task started after Pull-to-HUD bridge
	float RejoinCommitUntil;                              // committed horizontal rejoin direction
	float RejoinTurnSide;                                 // -1/+1 world horizontal turn side
	int RejoinTaskKind;                                   // continuity group for rejoin side
	float DefensiveTurnCommitUntil;                       // hold one defensive turn side briefly
	float DefensiveTurnSide;                              // -1/+1 world horizontal turn side
	int Phase;
	int EnemyPursuitType;                                // 0 unknown, 1 lag, 2 pure, 3 lead/WEZ
	int ControlZoneState;                                // 0 none, 1 approach, 2 established
	float ControlZoneDwell;
	float Node35Until;                                   // paper two-circle win -> extended-six bridge
	int Node35State;                                     // 0 none, 1 extended-six, 2 control-zone approach
	int TrackSubMode;                                    // 0 none, 1 blend, 2 pure, 3 lead, 4 snap, 5 alpha
	int VPPMode;                                         // 0 none, 1 lag, 2 pure, 3 lead, 4 blend
	int MyDamageBand;                                    // 0 none, otherwise active phase band 1..3
	int EnemyDamageBand;

	Vector3 PreviousMyLocation;
	Vector3 PreviousTargetLocation;
	Vector3 MyVelocity;
	Vector3 TargetVelocity;
	Vector3 MyAcceleration;
	Vector3 TargetAcceleration;
	Vector3 PreviousMyVelocity;
	Vector3 PreviousTargetVelocity;
	Vector3 PredictedTargetLocation;
	Vector3 TargetTurnCenter;
	float TargetTurnRadius_M;
	float TargetTurnRate_DegSec;
	bool TargetTurnCircleValid;

	float MyLosRate_DegSec;								// ownship LOS-angle rate
	float PreviousMyLos_Degree;
	float EnemyLosRate_DegSec;								// adversary LOS-angle rate
	float PreviousEnemyLos_Degree;

	Vector3 VPVelocity;										// target-tied VP velocity used by PN
	bool VPVelocityValid;									// true only for target-tied VP; enables PN
	Vector3 PreviousPNAnchor;
	int PreviousPNTaskKind;									// PN anchor continuity
	float VppBlendWeight;
	int PreviousVppTaskKind;
	bool HasPreviousPNAnchor;
	bool VelocityPointingController;
	int LastControllerModeUsed;                            // 0 heuristic nose-pointing, 1 acceleration/velocity-pointing
	bool TerminalNoseAimActive;                            // true when the final controller points the nose, not velocity
	bool PaperTerminalAccelerationActive;                  // true when the paper-flow terminal acceleration exception is active
	bool AlphaBiasApplied;                                 // true when velocity-to-nose alpha compensation changed VP this tick
	float AlphaBiasAngleDeg;
	float AlphaBiasWeight;
	int TrackSubReason;                                    // Track/ConeLeadTrack aim-point reason for log analysis
	int ThrottleReason;                                    // terminal throttle branch reason for log analysis
	float SignedVPErrorDeg;                                // signed horizontal nose-to-VP error
	float SignedTargetErrorDeg;                            // signed horizontal nose-to-target error
	float NoseToTargetSignedDeg;
	float NoseToVPSignedDeg;
	float TargetErrorHorizontalDeg;                        // signed horizontal nose-to-target error
	float TargetErrorVerticalDeg;                          // signed vertical nose-to-target error
	float VPErrorHorizontalDeg;                            // signed horizontal nose-to-VP error
	float VPErrorVerticalDeg;                              // signed vertical nose-to-VP error
	bool RollSaturated;
	bool PitchSaturated;
	bool RudderSaturated;
	int ActiveCompetitionTask;								//Last selected CompetitionTaskKind (-2: frozen baseline)
	int LockedManeuverTask;									//One/two-circle held until paper winning-cue angle
	float ManeuverTurnDegrees;
	float ManeuverStartTime;
	float LockedManeuverSide;
	bool PendingScissors;									//Neutral one/two-circle cue leads to BEM block 21
	Vector3 PreviousManeuverForward;
	Vector3 PreviousLevelForward;
	Vector3 PreviousTargetLevelForward;
	float PlanarTurnRate_DegSec;
	float PlanarTurnRateSigned_DegSec;
	float TargetPlanarTurnRateSigned_DegSec;
	int CircleDirectionRelation;                          // +1 same world turn, -1 opposite, 0 unresolved
	int LastManeuverCueOutcome;                           // ManeuverCueOutcome
	uint32_t EvaluatedConditionMask;							//Conditions reached this tick
	uint32_t TrueConditionMask;								//Reached conditions that returned SUCCESS
	float LastRollCommand;
	float LastPitchCommand;
	float LastRudderCommand;
	bool IsAimmingMode;


};
