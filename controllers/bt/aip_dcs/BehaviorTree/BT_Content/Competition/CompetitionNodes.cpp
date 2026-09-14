#include "CompetitionNodes.h"

#include <algorithm>
#include <cmath>

namespace
{
	const double DEG = 57.29577951308232;
	const double FT = 0.3048;
	const double KNOT = 0.514444;
	const double G = 9.80665;
	const double TWO_PI = 6.283185307179586;
	const double IDENTIFIED_CORNER_KCAS = 344.0;
	const double PAPER_SIMILAR_ENERGY_MARGIN_M = 600.0;

	enum EnemyPursuitClass
	{
		EPT_UNKNOWN = 0,
		EPT_LAG = 1,
		EPT_PURE = 2,
		EPT_LEAD = 3
	};

	enum VPPModeKind
	{
		VPP_NONE = 0,
		VPP_LAG = 1,
		VPP_PURE = 2,
		VPP_LEAD = 3,
		VPP_BLEND = 4
	};

	enum TrackSubModeKind
	{
		TRACK_NONE = 0,
		TRACK_BLEND = 1,
		TRACK_PURE = 2,
		TRACK_LEAD = 3,
		TRACK_SNAP = 4,
		TRACK_ALPHA = 5
	};

	enum TrackReasonKind
	{
		TRACK_REASON_NONE = 0,
		TRACK_REASON_BLEND = 1,
		TRACK_REASON_CONE_CENTER_SNAP = 2,
		TRACK_REASON_STRICT_CONE = 3,
		TRACK_REASON_TERMINAL_LEAD = 4,
		TRACK_REASON_PHASE1_LEAD_SNAP = 5,
		TRACK_REASON_LOW_SPEED_SNAP = 6,
		TRACK_REASON_PHASE1_PRESCORE = 7,
		TRACK_REASON_PHASE1_FINAL = 8,
		TRACK_REASON_PHASE1_FINE_PURE = 9,
		TRACK_REASON_ALPHA = 10,
		TRACK_REASON_LOW_ENERGY_FINE = 11,
		TRACK_REASON_HIGH_CLOSURE_VERTICAL = 12
	};

	enum ThrottleReasonKind
	{
		THROTTLE_REASON_DEFAULT = 0,
		THROTTLE_REASON_TERMINAL_BRAKE = 1,
		THROTTLE_REASON_TERMINAL_CORNER = 2,
		THROTTLE_REASON_FINE_TRACK_ENERGY = 3,
		THROTTLE_REASON_CLOSE_BRAKE = 4,
		THROTTLE_REASON_REACCEL = 5
	};

	double clampValue(double value, double low, double high)
	{
		return std::max(low, std::min(value, high));
	}

	BT_Geometry::Vector3 normalized(BT_Geometry::Vector3 value)
	{
		value.normalize();
		return value;
	}

	// Rodrigues rotation of v about a unit axis.
	BT_Geometry::Vector3 rotateAboutAxis(const BT_Geometry::Vector3& v,
		const BT_Geometry::Vector3& axisUnit, double angleRad)
	{
		const double c = std::cos(angleRad);
		const double s = std::sin(angleRad);
		return v * c + axisUnit.cross(v) * s + axisUnit * (axisUnit.dot(v) * (1.0 - c));
	}

	BT_Geometry::Vector3 direction(const BT_Geometry::Vector3& from, const BT_Geometry::Vector3& to,
		const BT_Geometry::Vector3& fallback)
	{
		BT_Geometry::Vector3 value = to - from;
		return value.length() < 0.01 ? fallback : normalized(value);
	}

	double calibratedManeuverSpeed(double tasMps, double kcasKt)
	{
		// A full EM chart is not required for identical-F-16 relative-energy
		// decisions. For rate-fight speed control, retain the measured F-16
		// sustained-turn optimum and convert its KCAS to the current local TAS
		// using the simulator-provided KCAS/KTAS ratio.
		if (tasMps > 50.0 && kcasKt > 100.0)
			return clampValue(tasMps * IDENTIFIED_CORNER_KCAS / kcasKt, 170.0, 280.0);
		return 420.0 * KNOT;
	}

	double maneuverSpeed(const CPPBlackBoard* bb)
	{
		return calibratedManeuverSpeed(bb->MySpeed_MS, bb->MyKCAS_KT);
	}

	double targetManeuverSpeed(const CPPBlackBoard* bb)
	{
		return calibratedManeuverSpeed(bb->TargetSpeed_MS, bb->TargetKCAS_KT);
	}

	bool hasHighManeuverEnergy(const CPPBlackBoard* bb)
	{
		// BEM block 16 asks whether ownship is in the high-energy region of its
		// own E-M chart.  It is an absolute maneuver-energy question; relative
		// energy is evaluated later by the one/two-circle winning-cue blocks.
		// Requiring a unilateral energy surplus here made every equal-energy
		// competition start choose one-circle, even when both aircraft were well
		// above the identified corner-speed band.  That is not the published
		// flow and, from an abeam 3-9 start, needlessly reverses away from a
		// pure-pursuit opponent.  In the absence of a full altitude-indexed E-M
		// chart, use the calibrated local-KCAS corner band as the paper-faithful
		// observable proxy.
		const bool speedInTurnBand = bb->MyKCAS_KT > 0.0f
			? bb->MyKCAS_KT >= 0.9 * IDENTIFIED_CORNER_KCAS
			: bb->MySpeed_MS >= 0.9 * maneuverSpeed(bb);
		return speedInTurnBand;
	}

	bool isCloseLowEnergyStalemate(const CPPBlackBoard* bb, double rangeFt, double neutralSec)
	{
		return bb->NeutralTime > neutralSec && rangeFt < 3000.0 &&
			std::abs(bb->ClosureRate_MS / FT) < 500.0 &&
			bb->MySpeed_MS < 0.8 * maneuverSpeed(bb) &&
			bb->TargetSpeed_MS < 0.8 * targetManeuverSpeed(bb);
	}

	void setCommand(CPPBlackBoard* bb, const BT_Geometry::Vector3& vp, ThrottleMode mode, double targetSpeed,
		const BT_Geometry::Vector3& vpVelocity, bool vpVelocityValid)
	{
		bb->VP_Cartesian = vp;
		bb->ThrottleCommandMode = mode;
		bb->TargetSpeedCommand_MS = static_cast<float>(targetSpeed);
		bb->VPVelocity = vpVelocity;
		bb->VPVelocityValid = vpVelocityValid;
	}

	int damageBand(double losDeg, double rangeFt, int phase)
	{
		const double cones[] = { 1.0, 2.0, 3.0 };
		const double ranges[] = { 3000.0, 3500.0, 4000.0 };
		const int maxIndex = static_cast<int>(clampValue(static_cast<double>(phase), 1.0, 3.0)) - 1;
		if (losDeg < cones[maxIndex] && rangeFt > 500.0 && rangeFt < ranges[maxIndex])
			return maxIndex + 1;
		return 0;
	}

	double damageRate(double losDeg, double rangeFt, int phase)
	{
		const double ranges[] = { 3000.0, 3500.0, 4000.0 };
		const double coeffs[] = { 1.0, 0.3, 0.1 };
		const int band = damageBand(losDeg, rangeFt, phase);
		if (band <= 0)
			return 0.0;
		const int index = band - 1;
		return ((ranges[index] - rangeFt) / (ranges[index] - 500.0)) * coeffs[index];
	}

	double phaseConeDeg(int phase)
	{
		return phase <= 1 ? 1.0 : (phase == 2 ? 2.0 : 3.0);
	}

	double phaseMaxRangeFt(int phase)
	{
		return phase <= 1 ? 3000.0 : (phase == 2 ? 3500.0 : 4000.0);
	}

	bool inRangeBand(double rangeFt, double maxRangeFt, double bufferFt = 0.0)
	{
		return rangeFt > 500.0 && rangeFt < maxRangeFt + bufferFt;
	}

	double nearConePotential(double losDeg, double rangeFt, int phase, double guardDeg, double rangeBufferFt)
	{
		const double cone = phaseConeDeg(phase);
		const double maxRange = phaseMaxRangeFt(phase);
		if (!inRangeBand(rangeFt, maxRange, rangeBufferFt))
			return 0.0;
		const double outerCone = cone + std::max(guardDeg, 0.1);
		if (losDeg >= outerCone)
			return 0.0;
		const double angular = clampValue((outerCone - losDeg) / std::max(outerCone, 0.1), 0.0, 1.0);
		const double rangeScore = clampValue((maxRange + rangeBufferFt - rangeFt) /
			std::max(maxRange + rangeBufferFt - 500.0, 1.0), 0.0, 1.0);
		const double strict = damageRate(losDeg, rangeFt, phase);
		return std::max(strict, angular * rangeScore);
	}

	bool weakOpeningEdgeShot(const CPPBlackBoard* bb, double rangeFt, double maxRangeFt, double coneDeg)
	{
		const double openingFtps = std::max(0.0, static_cast<double>(bb->ClosureRate_MS) / FT);
		return bb->BFM == OBFM &&
			bb->Phase <= 1 &&
			rangeFt > maxRangeFt - 120.0 &&
			rangeFt < maxRangeFt + 520.0 &&
			openingFtps > 55.0 &&
			bb->MyKCAS_KT + 22.0f < bb->TargetKCAS_KT &&
			bb->Los_Degree < coneDeg + 1.5 &&
			bb->Los_Degree_Target > 60.0f &&
			damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0;
	}

	bool enemyHasBetterNearCone(const CPPBlackBoard* bb, double rangeFt, double coneDeg,
		double maxRangeFt, double nearGuardDeg, double marginDeg = 1.5)
	{
		const double enemyNearDeg = coneDeg + nearGuardDeg;
		const bool enemyActual = damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) > 0.0;
		const bool enemyNear = bb->Los_Degree_Target < enemyNearDeg &&
			inRangeBand(rangeFt, maxRangeFt, 600.0) &&
			bb->EnemyLosRate_DegSec < 1.0f;
		const bool enemyClearlyBetter =
			bb->Los_Degree_Target + marginDeg < bb->Los_Degree;
		return enemyActual || (enemyNear && enemyClearlyBetter);
	}

	// The opening head-on, still outside the WEZ, where ownship is roughly nose-on
	// and closing very fast. Only here is the defensive flow suppressed, so the
	// tree presses the alpha-bias gun shot through the merge instead of breaking
	// early. Bounds are deliberately narrow: widening them (suppressing defense
	// across the whole fast pass) measured 5 losses vs 1, because the defensive
	// turn/jink DOES break the bandit's gun solution in this model - defense is
	// valuable and must not be suppressed once inside the WEZ.
	bool isFastHeadOnMerge(const CPPBlackBoard* bb, double rangeFt)
	{
		const double maxRangeFt = phaseMaxRangeFt(bb->Phase);
		return bb->BFM == HABFM &&
			bb->Los_Degree > 3.0f &&
			bb->Los_Degree < 15.0f &&
			bb->Los_Degree_Target < 15.0f &&
			rangeFt > maxRangeFt &&
			bb->RunningTime < 5.0 &&
			bb->ClosureRate_MS / FT < -1000.0f;
	}

	bool isNeutralHeadOnRemerge(const CPPBlackBoard* bb, double rangeFt)
	{
		return bb->BFM == HABFM &&
			rangeFt > 1500.0 &&
			rangeFt < 5000.0 &&
			bb->Los_Degree > 12.0f &&
			bb->Los_Degree < 30.0f &&
			bb->Los_Degree_Target > 12.0f &&
			bb->Los_Degree_Target < 30.0f &&
			bb->ClosureRate_MS / FT < -500.0f;
	}

	bool predictedThreatCue(const CPPBlackBoard* bb, double rangeFt, double coneDeg, double maxRangeFt,
		double enemyLosDeg, double rangeBufFt, double friendlyLosDeg, double minMyLosDeg,
		double maxEnemyLosRateDegps, double enemyBetterMarginDeg, double potentialMargin,
		double neutralSec, double stalemateRangeFt)
	{
		const int phase = bb->RunningTime < 100.0 ? 1 : (bb->RunningTime < 150.0 ? 2 : 3);
		const bool closeNeutralStalemate =
			bb->BFM == HABFM &&
			bb->NeutralTime > neutralSec &&
			rangeFt < stalemateRangeFt;
		const double gateDeg = clampValue(enemyLosDeg, coneDeg + 1.0, coneDeg + 8.0);
		const double myPotential = nearConePotential(bb->Los_Degree, rangeFt, phase,
			gateDeg - coneDeg + 2.0, rangeBufFt);
		const double enemyPotential = nearConePotential(bb->Los_Degree_Target, rangeFt, phase,
			gateDeg - coneDeg, rangeBufFt);
		const bool ownActualGun = damageRate(bb->Los_Degree, rangeFt, phase) > 0.0;
		const bool ownNearCone = bb->Los_Degree < friendlyLosDeg &&
			inRangeBand(rangeFt, maxRangeFt, rangeBufFt);
		const bool enemyActual = damageRate(bb->Los_Degree_Target, rangeFt, phase) > 0.0;
		const bool enemyNear = bb->Los_Degree_Target < gateDeg &&
			inRangeBand(rangeFt, maxRangeFt, 600.0) &&
			bb->EnemyLosRate_DegSec < 1.0f;
		const bool enemyClearlyBetter =
			bb->Los_Degree_Target + enemyBetterMarginDeg < bb->Los_Degree;
		const bool enemyBetter = enemyActual || (enemyNear && enemyClearlyBetter);
		const bool enemyPotentialDominant =
			enemyBetter ||
			(enemyPotential > myPotential + potentialMargin &&
			 bb->Los_Degree > friendlyLosDeg);
		return !ownActualGun &&
			!closeNeutralStalemate &&
			!isFastHeadOnMerge(bb, rangeFt) &&
			!isNeutralHeadOnRemerge(bb, rangeFt) &&
			bb->Los_Degree > minMyLosDeg &&
			bb->Los_Degree_Target < gateDeg &&
			inRangeBand(rangeFt, maxRangeFt, rangeBufFt) &&
			bb->EnemyLosRate_DegSec < maxEnemyLosRateDegps &&
			enemyPotentialDominant &&
			(!ownNearCone || enemyBetter);
	}

	int classifyEnemyPursuit(const CPPBlackBoard* bb, double rangeFt, int phase)
	{
		const double coneDeg = phaseConeDeg(phase);
		const double maxRangeFt = phaseMaxRangeFt(phase);
		const double closingFtps = std::max(0.0, -static_cast<double>(bb->ClosureRate_MS) / FT);
		const double absClosureFtps = std::abs(static_cast<double>(bb->ClosureRate_MS) / FT);
		const bool enemyActualGun = damageRate(bb->Los_Degree_Target, rangeFt, phase) > 0.0;
		const bool enemyNearGun = bb->Los_Degree_Target < coneDeg + 3.0 &&
			rangeFt > 500.0 && rangeFt < maxRangeFt + 800.0 &&
			std::abs(static_cast<double>(bb->EnemyLosRate_DegSec)) < 1.5;
		if (enemyActualGun || enemyNearGun)
			return EPT_LEAD;
		const bool purePursuit =
			bb->Los_Degree_Target < 18.0f &&
			std::abs(static_cast<double>(bb->EnemyLosRate_DegSec)) < 8.0 &&
			(closingFtps > 150.0 || rangeFt < maxRangeFt + 1800.0);
		if (purePursuit)
			return EPT_PURE;
		const bool lagPursuit =
			bb->Los_Degree_Target > 18.0f &&
			bb->Los_Degree_Target < 65.0f &&
			absClosureFtps < 900.0;
		if (lagPursuit)
			return EPT_LAG;
		return EPT_UNKNOWN;
	}

	void updateControlZoneState(CPPBlackBoard* bb, double rangeFt, bool enemyActualGun)
	{
		if (bb->BFM != OBFM || enemyActualGun)
		{
			bb->ControlZoneState = 0;
			bb->ControlZoneDwell = 0.0f;
			if (bb->RunningTime >= bb->Node35Until)
				bb->Node35State = 0;
			return;
		}

		const double absClosureFtps = std::abs(static_cast<double>(bb->ClosureRate_MS) / FT);
		const bool approach =
			rangeFt > 1400.0 && rangeFt < 5200.0 &&
			bb->Los_Degree < 75.0f &&
			bb->Los_Degree_Target > 40.0f &&
			absClosureFtps < 1600.0;
		const bool establishedEntry =
			rangeFt > 1300.0 && rangeFt < 3600.0 &&
			bb->Los_Degree < 35.0f &&
			bb->Los_Degree_Target > 70.0f &&
			absClosureFtps < 500.0;
		const bool establishedStay =
			rangeFt > 900.0 && rangeFt < 4200.0 &&
			bb->Los_Degree < 45.0f &&
			bb->Los_Degree_Target > 55.0f &&
			absClosureFtps < 800.0;

		if (bb->ControlZoneState == 2 && establishedStay)
			bb->ControlZoneDwell = static_cast<float>(std::min(2.0,
				static_cast<double>(bb->ControlZoneDwell) + std::max(bb->DeltaSecond, 1e-3)));
		else if (establishedEntry)
			bb->ControlZoneDwell = static_cast<float>(
				static_cast<double>(bb->ControlZoneDwell) + std::max(bb->DeltaSecond, 1e-3));
		else
			bb->ControlZoneDwell = 0.0f;

		if (bb->ControlZoneDwell > 0.30f)
			bb->ControlZoneState = 2;
		else if (approach)
			bb->ControlZoneState = 1;
		else
			bb->ControlZoneState = 0;

		if (bb->ControlZoneState > 0 && bb->Node35State == 1)
			bb->Node35State = 2;
		if (bb->RunningTime >= bb->Node35Until && bb->Node35State != 2)
			bb->Node35State = 0;
	}

	void armNode35(CPPBlackBoard* bb, double seconds)
	{
		bb->Node35Until = static_cast<float>(std::max(
			static_cast<double>(bb->Node35Until), bb->RunningTime + seconds));
		bb->Node35State = 1;
		bb->VppBlendWeight = std::min(bb->VppBlendWeight, 0.35f);
	}

	BT_Geometry::Vector3 levelForward(const CPPBlackBoard* bb)
	{
		BT_Geometry::Vector3 forward = bb->MyVelocity.length() > 1.0
			? bb->MyVelocity
			: bb->MyForwardVector;
		forward.Z = 0.0;
		if (forward.length() < 0.01)
			forward = BT_Geometry::Vector3(1, 0, 0);
		return normalized(forward);
	}

	BT_Geometry::Vector3 levelRight(const CPPBlackBoard* bb)
	{
		return normalized(BT_Geometry::Vector3(0, 0, 1).cross(levelForward(bb)));
	}

	// Altitude (ft) needed to pull level from the current dive: pull-up arc loss
	// (radius from speed and the identified 6.5-g authority) plus a sink-rate
	// lead term, above the hard-deck floor and a safety margin. Shared by the
	// candidate's GroundDanger gate and the baseline so both recover the same way.
	double groundRecoveryAltitudeFt(const CPPBlackBoard* bb, double floorFt, double marginFt)
	{
		const double speed = std::max(bb->MyVelocity.length(), 1.0);
		const double gamma = std::asin(clampValue(bb->MyVelocity.Z / speed, -1.0, 1.0));
		const double availableNz = 6.5;
		const double denominator = G * std::max(availableNz - std::cos(gamma), 0.5);
		const double pullRadius = speed * speed / denominator;
		const double recoveryTurnFt = (pullRadius * (1.0 - std::cos(std::abs(gamma)))) / FT;
		const double sinkFtPerSec = std::max(0.0, -static_cast<double>(bb->AltSpeed)) / FT;
		return floorFt + marginFt + recoveryTurnFt + sinkFtPerSec * 0.5;
	}

	// Pull toward level flight and climb. Using level-forward (not the body
	// forward axis) avoids re-aiming into the ground during a steep dive.
	BT_Geometry::Vector3 groundRecoveryVP(const CPPBlackBoard* bb)
	{
		return bb->MyLocation_Cartesian + levelForward(bb) * 4000.0 + BT_Geometry::Vector3(0, 0, 3000.0);
	}

	double turnSide(const CPPBlackBoard* bb)
	{
		BT_Geometry::Vector3 toTarget = direction(bb->MyLocation_Cartesian, bb->TargetLocaion_Cartesian,
			bb->MyForwardVector);
		return levelForward(bb).cross(toTarget).Z >= 0.0 ? 1.0 : -1.0;
	}

	bool hostileVerticalUp(const CPPBlackBoard* bb)
	{
		return bb->TargetVelocity.Z > 20.0 && bb->TargetForwardVector.Z > 0.12;
	}

	bool hostileVerticalDown(const CPPBlackBoard* bb)
	{
		return bb->TargetVelocity.Z < -20.0 && bb->TargetForwardVector.Z < -0.12;
	}

	double targetTurnSide(const CPPBlackBoard* bb)
	{
		BT_Geometry::Vector3 targetForward = bb->TargetVelocity.length() > 1.0
			? bb->TargetVelocity
			: bb->TargetForwardVector;
		targetForward.Z = 0.0;
		if (targetForward.length() < 0.01)
			targetForward = bb->TargetForwardVector;
		targetForward = normalized(targetForward);

		// Prefer the established target turn. Before its turn develops, assume
		// it will turn toward ownship, which is the opponent response used by
		// the neutral-merge BEM geometry.
		if (bb->TargetTurnCircleValid)
		{
			BT_Geometry::Vector3 targetAcceleration = bb->TargetAcceleration;
			targetAcceleration.Z = 0.0;
			const double observedTurn = targetForward.cross(targetAcceleration).Z;
			if (std::abs(observedTurn) > 0.01)
				return observedTurn >= 0.0 ? 1.0 : -1.0;
		}
		BT_Geometry::Vector3 toOwnship = direction(
			bb->TargetLocaion_Cartesian, bb->MyLocation_Cartesian, -bb->MyForwardVector);
		toOwnship.Z = 0.0;
		return targetForward.cross(toOwnship).Z >= 0.0 ? 1.0 : -1.0;
	}


	void resetManeuverLock(CPPBlackBoard* bb)
	{
		bb->LockedManeuverTask = -1;
		bb->ManeuverTurnDegrees = 0.0f;
		bb->ManeuverStartTime = 0.0f;
		bb->LockedManeuverSide = 0.0f;
		bb->PreviousManeuverForward = BT_Geometry::Vector3(0, 0, 0);
	}

	void startManeuverLock(CPPBlackBoard* bb, int taskKind, double side)
	{
		bb->LockedManeuverTask = taskKind;
		bb->ManeuverTurnDegrees = 0.0f;
		bb->ManeuverStartTime = static_cast<float>(bb->RunningTime);
		bb->LockedManeuverSide = static_cast<float>(side >= 0.0 ? 1.0 : -1.0);
		bb->PreviousManeuverForward = levelForward(bb);
	}

	double signedPlanarAngleDeg(BT_Geometry::Vector3 from, BT_Geometry::Vector3 to)
	{
		from.Z = 0.0;
		to.Z = 0.0;
		if (from.length() < 0.01 || to.length() < 0.01)
			return 0.0;
		from.normalize();
		to.normalize();
		return std::atan2(from.cross(to).Z,
			clampValue(from.dot(to), -1.0, 1.0)) * DEG;
	}

	double planarAngleDeg(BT_Geometry::Vector3 from, BT_Geometry::Vector3 to)
	{
		return std::abs(signedPlanarAngleDeg(from, to));
	}

	double totalAngleDeg(BT_Geometry::Vector3 from, BT_Geometry::Vector3 to)
	{
		if (from.length() < 0.01 || to.length() < 0.01)
			return 0.0;
		from.normalize();
		to.normalize();
		return std::acos(clampValue(from.dot(to), -1.0, 1.0)) * DEG;
	}

	double signedVerticalErrorDeg(const BT_Geometry::Vector3& forward,
		const BT_Geometry::Vector3& up, const BT_Geometry::Vector3& target)
	{
		if (target.length() < 0.01 || up.length() < 0.01)
			return 0.0;
		BT_Geometry::Vector3 targetUnit = target;
		BT_Geometry::Vector3 upUnit = up;
		targetUnit.normalize();
		upUnit.normalize();
		// Body-frame elevation is exactly the arcsine of the target direction's
		// projection on body-up.  The previous sqrt(total^2-horizontal^2)
		// small-angle approximation collapsed to zero during banked turns and
		// lost the sign precisely at fixed-gun capture.
		return std::asin(clampValue(upUnit.dot(targetUnit), -1.0, 1.0)) * DEG;
	}

	double committedRejoinSide(CPPBlackBoard* bb,
		const BT_Geometry::Vector3& desiredPoint, int continuityGroup, double commitSec)
	{
		const BT_Geometry::Vector3 forward = levelForward(bb);
		BT_Geometry::Vector3 desired = desiredPoint - bb->MyLocation_Cartesian;
		desired.Z = 0.0;
		if (desired.length() < 1.0)
			desired = forward;
		else
			desired.normalize();
		const double crossZ = forward.cross(desired).Z;
		const double dot = clampValue(forward.dot(desired), -1.0, 1.0);
		const bool newCommit = bb->RejoinTaskKind != continuityGroup ||
			bb->RunningTime >= bb->RejoinCommitUntil || std::abs(bb->RejoinTurnSide) < 0.5f;
		if (newCommit)
		{
			double side = 0.0;
			if (std::abs(crossZ) > 0.02)
				side = crossZ >= 0.0 ? 1.0 : -1.0;
			else if (dot < 0.0)
				side = targetTurnSide(bb);
			else
				side = turnSide(bb);
			bb->RejoinTurnSide = static_cast<float>(side >= 0.0 ? 1.0 : -1.0);
			bb->RejoinTaskKind = continuityGroup;
			bb->RejoinCommitUntil = static_cast<float>(bb->RunningTime + std::max(commitSec, 1.0));
		}
		if (bb->Los_Degree < 50.0f && bb->Distance / FT < 5500.0)
			bb->RejoinCommitUntil = static_cast<float>(bb->RunningTime);
		return bb->RejoinTurnSide >= 0.0f ? 1.0 : -1.0;
	}

	double committedDefensiveSide(CPPBlackBoard* bb, double commitSec)
	{
		const bool newCommit = bb->RunningTime >= bb->DefensiveTurnCommitUntil ||
			std::abs(bb->DefensiveTurnSide) < 0.5f;
		if (newCommit)
		{
			const double side = turnSide(bb);
			bb->DefensiveTurnSide = static_cast<float>(side >= 0.0 ? 1.0 : -1.0);
			bb->DefensiveTurnCommitUntil = static_cast<float>(
				bb->RunningTime + std::max(commitSec, 0.8));
		}
		const double angularAdvantage =
			static_cast<double>(bb->Los_Degree_Target - bb->Los_Degree);
		if (bb->Los_Degree_Target > 105.0f && angularAdvantage > -5.0)
			bb->DefensiveTurnCommitUntil = static_cast<float>(bb->RunningTime);
		return bb->DefensiveTurnSide >= 0.0f ? 1.0 : -1.0;
	}

	BT_Geometry::Vector3 targetLevelForward(const CPPBlackBoard* bb)
	{
		BT_Geometry::Vector3 forward = bb->TargetVelocity.length() > 1.0
			? bb->TargetVelocity
			: bb->TargetForwardVector;
		forward.Z = 0.0;
		if (forward.length() < 0.01)
			forward = levelForward(bb);
		return normalized(forward);
	}

	BT_Geometry::Vector3 targetLevelRight(const CPPBlackBoard* bb)
	{
		BT_Geometry::Vector3 right = BT_Geometry::Vector3(0, 0, 1).cross(targetLevelForward(bb));
		if (right.length() < 0.01)
			right = levelRight(bb);
		return normalized(right);
	}

	double enemyConeEscapeSide(CPPBlackBoard* bb, double fallbackSide)
	{
		const BT_Geometry::Vector3 targetRight = targetLevelRight(bb);
		const BT_Geometry::Vector3 relativePosition =
			bb->MyLocation_Cartesian - bb->TargetLocaion_Cartesian;
		const double lateralOffset = targetRight.dot(relativePosition);
		if (std::abs(lateralOffset) > 5.0)
			return lateralOffset >= 0.0 ? 1.0 : -1.0;
		const BT_Geometry::Vector3 relativeVelocity = bb->MyVelocity - bb->TargetVelocity;
		const double lateralRate = targetRight.dot(relativeVelocity);
		if (std::abs(lateralRate) > 0.5)
			return lateralRate >= 0.0 ? 1.0 : -1.0;
		return fallbackSide >= 0.0 ? 1.0 : -1.0;
	}

	double defensiveVerticalSide(const CPPBlackBoard* bb)
	{
		const double altitudeDifference =
			bb->MyLocation_Cartesian.Z - bb->TargetLocaion_Cartesian.Z;
		if (std::abs(altitudeDifference) > 200.0 * FT)
			return altitudeDifference >= 0.0 ? 1.0 : -1.0;
		// With equal starts, prefer an upward out-of-plane break; GroundDanger
		// already owns hard-deck protection before DBFM is evaluated.
		return 1.0;
	}

	BT_Geometry::Vector3 enemyConeEscapeVP(CPPBlackBoard* bb, double side,
		double lateralMeters, double forwardMeters, double verticalMeters)
	{
		const BT_Geometry::Vector3 targetRight = targetLevelRight(bb);
		const double escapeSide = enemyConeEscapeSide(bb, side);
		const double verticalSide = defensiveVerticalSide(bb);
		return bb->MyLocation_Cartesian +
			levelForward(bb) * forwardMeters +
			targetRight * (escapeSide * lateralMeters) +
			BT_Geometry::Vector3(0, 0, verticalSide * verticalMeters);
	}

	double sustainedTurnRadius(double speedMps)
	{
		// OBFM turn-circle geometry should match the 9G command authority used
		// by Controller_CY; the 6.5G value is kept only for ground recovery.
		const double availableNz = 9.0;
		return speedMps * speedMps / (G * std::max(availableNz - 1.0, 0.5));
	}

	BT_Geometry::Vector3 projectedInPlane(const BT_Geometry::Vector3& vector,
		const BT_Geometry::Vector3& axisUnit)
	{
		return vector - axisUnit * vector.dot(axisUnit);
	}

	BT_Geometry::Vector3 limitedVector(BT_Geometry::Vector3 vector, double maxLength)
	{
		const double length = vector.length();
		if (length <= maxLength || length < 1e-3)
			return vector;
		return vector * (maxLength / length);
	}

	BT_Geometry::Vector3 targetTurnAxis(const CPPBlackBoard* bb)
	{
		if (bb->TargetTurnCircleValid)
		{
			BT_Geometry::Vector3 radial = bb->TargetLocaion_Cartesian - bb->TargetTurnCenter;
			BT_Geometry::Vector3 axis = radial.cross(bb->TargetVelocity);
			if (axis.length() > 1.0)
				return normalized(axis);
		}
		return BT_Geometry::Vector3(0, 0, 1);
	}

	BT_Geometry::Vector3 turnPlaneLateral(const CPPBlackBoard* bb, double side,
		const BT_Geometry::Vector3& forward)
	{
		BT_Geometry::Vector3 lateral = levelRight(bb) * side;
		const BT_Geometry::Vector3 axis = targetTurnAxis(bb);
		BT_Geometry::Vector3 planeLateral = axis.cross(forward);
		if (planeLateral.length() > 0.1)
		{
			planeLateral.normalize();
			lateral = planeLateral.dot(lateral) >= 0.0 ? planeLateral : -planeLateral;
		}
		return lateral;
	}

	BT_Geometry::Vector3 virtualLagVPP(const CPPBlackBoard* bb, double desiredSpeed,
		double side, double kLat, double maxCorrectionM)
	{
		const BT_Geometry::Vector3 axis = targetTurnAxis(bb);
		const BT_Geometry::Vector3 forward = levelForward(bb);
		const BT_Geometry::Vector3 ownInside = turnPlaneLateral(bb, side, forward);
		const double ownRadius = clampValue(sustainedTurnRadius(std::max(desiredSpeed, 80.0)), 120.0, 2200.0);
		const BT_Geometry::Vector3 ownVirtualCenter = bb->MyLocation_Cartesian + ownInside * ownRadius;

		BT_Geometry::Vector3 targetCenter = bb->TargetTurnCenter;
		if (!bb->TargetTurnCircleValid)
		{
			BT_Geometry::Vector3 targetForward = bb->TargetVelocity.length() > 1.0
				? bb->TargetVelocity
				: bb->TargetForwardVector;
			targetForward = projectedInPlane(targetForward, axis);
			if (targetForward.length() < 0.1)
				targetForward = bb->TargetForwardVector;
			targetForward = normalized(targetForward);
			BT_Geometry::Vector3 targetInside = BT_Geometry::Vector3(0, 0, 1).cross(targetForward) * targetTurnSide(bb);
			if (targetInside.length() < 0.1)
				targetInside = ownInside;
			else
				targetInside.normalize();
			const double targetRadius = clampValue(
				sustainedTurnRadius(std::max(static_cast<double>(bb->TargetSpeed_MS), 80.0)),
				120.0, 2500.0);
			targetCenter = bb->TargetLocaion_Cartesian + targetInside * targetRadius;
		}

		BT_Geometry::Vector3 centerError = projectedInPlane(targetCenter - ownVirtualCenter, axis);
		centerError = limitedVector(centerError, maxCorrectionM);

		// You & Shim LagVPP: align attacker/target turn centers horizontally,
		// then replace the vertical component with energy exchange. Positive Z
		// is up here, so excess speed moves the VPP above the target.
		BT_Geometry::Vector3 vpp = bb->TargetLocaion_Cartesian + centerError * kLat;
		const double speedEnergy = (bb->MySpeed_MS * bb->MySpeed_MS - desiredSpeed * desiredSpeed) / (2.0 * G);
		const double climbRateEnergy = (bb->TargetVelocity.Z - bb->MyVelocity.Z) * 2.0;
		vpp.Z = bb->TargetLocaion_Cartesian.Z +
			clampValue(speedEnergy + climbRateEnergy, -900.0, 2200.0);
		return vpp;
	}

	BT_Geometry::Vector3 smoothedVPP(CPPBlackBoard* bb, const BT_Geometry::Vector3& lagVpp,
		const BT_Geometry::Vector3& pure, const BT_Geometry::Vector3& lead, double desiredWeight)
	{
		const double maxStep = clampValue(0.9 * std::max(static_cast<double>(bb->DeltaSecond), 1e-3), 0.01, 0.06);
		const double current = clampValue(bb->VppBlendWeight, 0.0, 1.0);
		const double next = current + clampValue(desiredWeight - current, -maxStep, maxStep);
		bb->VppBlendWeight = static_cast<float>(clampValue(next, 0.0, 1.0));
		const double w = bb->VppBlendWeight;
		if (std::abs(w - 0.50) < 0.08)
			bb->VPPMode = VPP_PURE;
		else if (w < 0.35)
			bb->VPPMode = VPP_LAG;
		else if (w > 0.65)
			bb->VPPMode = VPP_LEAD;
		else
			bb->VPPMode = VPP_BLEND;
		if (w <= 0.5)
		{
			const double pureWeight = w / 0.5;
			return lagVpp * (1.0 - pureWeight) + pure * pureWeight;
		}
		const double leadWeight = (w - 0.5) / 0.5;
		return pure * (1.0 - leadWeight) + lead * leadWeight;
	}

	// APG alpha-bias (spec 2.2): the acceleration controller aligns the
	// VELOCITY vector with VP, but the damage cone follows the NOSE which sits
	// AoA above the velocity vector. Rotate the aim point down by alpha about
	// the pitch axis, blended in as the nose approaches the damage cone.
	BT_Geometry::Vector3 alphaBiasedAim(CPPBlackBoard* bb, const BT_Geometry::Vector3& vp,
		const BT_Geometry::Vector3& rightAxis, bool allowVelocityAlphaBias = true)
	{
		if (!allowVelocityAlphaBias || !bb->VelocityPointingController || bb->MyVelocity.length() <= 1.0)
			return vp;
		BT_Geometry::Vector3 velocityDir = normalized(bb->MyVelocity);
		const double sinAlpha = -(velocityDir.cross(bb->MyForwardVector)).dot(rightAxis);
		const double cosAlpha = velocityDir.dot(bb->MyForwardVector);
		const double alpha = std::atan2(sinAlpha, cosAlpha);
		bb->AlphaBiasAngleDeg = static_cast<float>(alpha * DEG);
		BT_Geometry::Vector3 aim = vp - bb->MyLocation_Cartesian;
		const double aimDistance = aim.length();
		if (aimDistance <= 1.0 || std::abs(alpha) >= 0.35)
			return vp;
		BT_Geometry::Vector3 aimDir = rotateAboutAxis(aim / aimDistance, rightAxis, alpha);
		BT_Geometry::Vector3 vpApg = bb->MyLocation_Cartesian + aimDir * aimDistance;
		const double coneDeg = phaseConeDeg(bb->Phase);
		// VER08: the acceleration controller steers the velocity vector, while
		// scoring uses the nose vector. Begin AoA compensation before the final
		// 1/2/3-deg cone instead of waiting until LOS < 2*cone, which left the
		// phase-1 traces stalled around 4-8 deg.
		const double activationDeg = std::max(20.0, 6.0 * coneDeg);
		const double w = clampValue((activationDeg - bb->Los_Degree) / activationDeg, 0.0, 1.0);
		bb->AlphaBiasWeight = static_cast<float>(w);
		bb->AlphaBiasApplied = w > 1.0e-4;
		return vp * (1.0 - w) + vpApg * w;
	}

	BT_Geometry::Vector3 limitedTurnAim(const CPPBlackBoard* bb,
		const BT_Geometry::Vector3& desiredPoint, double maxTurnDeg, double aimDistanceM)
	{
		BT_Geometry::Vector3 velocityDir = bb->MyVelocity.length() > 1.0
			? normalized(bb->MyVelocity)
			: bb->MyForwardVector;
		BT_Geometry::Vector3 desiredDir = direction(
			bb->MyLocation_Cartesian, desiredPoint, velocityDir);
		const double angle = velocityDir.angleBetween(desiredDir);
		const double maxAngle = std::max(1.0, maxTurnDeg) / DEG;
		BT_Geometry::Vector3 commandDir = desiredDir;
		if (angle > maxAngle)
		{
			commandDir.sLerp(velocityDir, desiredDir,
				maxAngle / std::max(angle, 1.0e-6), true);
			commandDir.normalize();
		}
		return bb->MyLocation_Cartesian + commandDir * std::max(aimDistanceM, 1000.0);
	}

	// A target directly behind the velocity vector is a pursuit-guidance
	// singularity: Vhat x LOS approaches zero and a target-tied far VP may
	// produce almost no horizontal bank.  Build an ownship-relative, level
	// rejoin point that commands a bounded horizontal turn toward the target
	// while preserving energy.  It is intentionally not PN/target tied.
	BT_Geometry::Vector3 horizontalRejoinVP(const CPPBlackBoard* bb,
		const BT_Geometry::Vector3& desiredPoint, double maxTurnDeg,
		double aimDistanceM, double maxVerticalM, double committedSide)
	{
		const BT_Geometry::Vector3 worldUp(0, 0, 1);
		const BT_Geometry::Vector3 forward = levelForward(bb);
		BT_Geometry::Vector3 desiredHorizontal = desiredPoint - bb->MyLocation_Cartesian;
		desiredHorizontal.Z = 0.0;
		if (desiredHorizontal.length() < 1.0)
			desiredHorizontal = forward;
		else
			desiredHorizontal.normalize();
		const double dot = clampValue(forward.dot(desiredHorizontal), -1.0, 1.0);
		const double crossZ = forward.cross(desiredHorizontal).Z;
		double signedAngle = std::atan2(crossZ, dot);
		const double unsignedAngle = std::acos(dot);
		// Hold a single world turn side while the target is behind or the fight
		// is far. Recomputing atan2 every tick caused left/right chatter and the
		// 10-30 kft rejoin orbit seen in VER09.
		if ((dot < 0.55 || (bb->Distance / FT > 6500.0 && unsignedAngle > 25.0 / DEG)) &&
			std::abs(committedSide) > 0.5)
			signedAngle = (committedSide >= 0.0 ? 1.0 : -1.0) * unsignedAngle;
		else if (std::abs(crossZ) < 1.0e-5 && dot < 0.0)
			signedAngle = (committedSide >= 0.0 ? 1.0 : -1.0) * 3.14159265358979323846;
		const double maxAngle = std::max(5.0, maxTurnDeg) / DEG;
		const double commandedAngle = clampValue(signedAngle, -maxAngle, maxAngle);
		BT_Geometry::Vector3 commandDir =
			rotateAboutAxis(forward, worldUp, commandedAngle);
		commandDir.Z = 0.0;
		commandDir.normalize();
		const double vertical = clampValue(
			desiredPoint.Z - bb->MyLocation_Cartesian.Z,
			-maxVerticalM, maxVerticalM);
		return bb->MyLocation_Cartesian +
			commandDir * std::max(aimDistanceM, 2500.0) + worldUp * vertical;
	}

	BT_Geometry::Vector3 predictedTargetAt(const CPPBlackBoard* bb, double predictSec)
	{
		predictSec = clampValue(predictSec, 0.05, 1.25);
		BT_Geometry::Vector3 radial = bb->TargetLocaion_Cartesian - bb->TargetTurnCenter;
		BT_Geometry::Vector3 axis = radial.cross(bb->TargetVelocity);
		if (bb->TargetTurnCircleValid && axis.length() > 1.0 && radial.length() > 1.0)
		{
			axis.normalize();
			const double turnRate = bb->TargetVelocity.length() /
				std::max(static_cast<double>(bb->TargetTurnRadius_M), 1.0);
			return bb->TargetTurnCenter + rotateAboutAxis(radial, axis, turnRate * predictSec);
		}

		// Competition LeadVPP is not a bullet-impact point. It is a short-horizon
		// future cone-alignment point that compensates target crossing and nose/
		// controller lag. Clamp acceleration so noisy finite differences do not
		// throw the VP outside the fight.
		const BT_Geometry::Vector3 acceleration = limitedVector(bb->TargetAcceleration, 45.0);
		return bb->TargetLocaion_Cartesian + bb->TargetVelocity * predictSec +
			acceleration * (0.5 * predictSec * predictSec);
	}

	BT_Geometry::Vector3 angularlyLimitedLeadAim(const CPPBlackBoard* bb,
		const BT_Geometry::Vector3& desiredLead, double maxLeadDeg)
	{
		BT_Geometry::Vector3 pureDir = direction(bb->MyLocation_Cartesian,
			bb->TargetLocaion_Cartesian, bb->MyForwardVector);
		BT_Geometry::Vector3 leadDir = direction(bb->MyLocation_Cartesian,
			desiredLead, pureDir);
		const double angle = pureDir.angleBetween(leadDir);
		const double maxAngle = std::max(0.25, maxLeadDeg) / DEG;
		if (angle <= maxAngle)
			return desiredLead;
		BT_Geometry::Vector3 limitedDir;
		limitedDir.sLerp(pureDir, leadDir, maxAngle / std::max(angle, 1.0e-6), true);
		limitedDir.normalize();
		const double distance = std::max(bb->MyLocation_Cartesian.distance(desiredLead), 1.0);
		return bb->MyLocation_Cartesian + limitedDir * distance;
	}

	BT_Geometry::Vector3 snapThroughGunAim(const CPPBlackBoard* bb,
		const BT_Geometry::Vector3& aimPoint, double gain, double maxExtraDeg)
	{
		BT_Geometry::Vector3 forward = bb->MyForwardVector;
		if (forward.length() < 0.1)
			return aimPoint;
		forward.normalize();
		const BT_Geometry::Vector3 aimDir = direction(bb->MyLocation_Cartesian, aimPoint, forward);
		BT_Geometry::Vector3 axis = forward.cross(aimDir);
		const double axisLength = axis.length();
		if (axisLength < 1.0e-4)
			return aimPoint;
		axis = axis / axisLength;
		const double angle = std::acos(clampValue(forward.dot(aimDir), -1.0, 1.0));
		const double snapAngle = std::min(angle * std::max(gain, 1.0),
			angle + std::max(maxExtraDeg, 0.0) / DEG);
		const BT_Geometry::Vector3 snapDir = rotateAboutAxis(forward, axis, snapAngle);
		const double distance = std::max(bb->MyLocation_Cartesian.distance(aimPoint), 1000.0);
		return bb->MyLocation_Cartesian + snapDir * distance;
	}

	double apparentLosRateDegSec(const CPPBlackBoard* bb)
	{
		const BT_Geometry::Vector3 line = direction(bb->MyLocation_Cartesian,
			bb->TargetLocaion_Cartesian, bb->MyForwardVector);
		const BT_Geometry::Vector3 relativeVelocity = bb->TargetVelocity - bb->MyVelocity;
		const BT_Geometry::Vector3 lateralVelocity = relativeVelocity - line * relativeVelocity.dot(line);
		const double geometricRate = bb->Distance > 1.0 ? lateralVelocity.length() / bb->Distance * DEG : 0.0;
		return std::max(std::abs(static_cast<double>(bb->MyLosRate_DegSec)), geometricRate);
	}
}

namespace Action
{
	CompetitionNode::CompetitionNode(const std::string& name, const NodeConfiguration& config) : SyncActionNode(name, config) {}

	PortsList CompetitionNode::providedPorts()
	{
		return {
			InputPort<CPPBlackBoard*>("BB"),
			InputPort<double>("MarginFt"), InputPort<double>("MarginM"), InputPort<double>("FloorFt"), InputPort<double>("EnterDeg"),
			InputPort<double>("ExitDeg"), InputPort<double>("GuardDeg"), InputPort<double>("EnemyLOSDeg"),
			InputPort<double>("RangeBufFt"), InputPort<double>("RangeFt"), InputPort<double>("ClosureFtps"),
			InputPort<double>("OpeningFtps"),
			InputPort<double>("AngleOffDeg"), InputPort<double>("HoldRangeFt"), InputPort<double>("LagBlendDeg"),
			InputPort<bool>("AlphaBias"), InputPort<bool>("ConeCenterSnap"),
			InputPort<bool>("CloseNoCue"), InputPort<bool>("BreakJinkLock"),
			InputPort<bool>("ForceLiteralHUD"),
			InputPort<double>("SnapGain"), InputPort<double>("SnapExtraDeg"),
			InputPort<double>("EntryArcDeg"), InputPort<double>("MinPhase"),
			InputPort<double>("LeadThreshold"), InputPort<double>("StandoffBufFt"), InputPort<double>("TTMSec"),
			InputPort<double>("OffsetFt"), InputPort<double>("LeadTurnTTMSec"),
			InputPort<double>("AcceptHeadonIfDiffBelow"), InputPort<double>("NeutralSec"),
			InputPort<double>("StalemateRangeFt"),
			InputPort<double>("JinkPeriodSec"), InputPort<double>("JinkCommitSec"), InputPort<double>("KpT"), InputPort<double>("KdT"),
			InputPort<double>("DThrMaxPerTick"), InputPort<double>("MinSec"),
			InputPort<double>("MinLOSDeg"), InputPort<double>("MaxLOSDeg"), InputPort<double>("MinMyLOSDeg"),
			InputPort<double>("MaxMyLOSDeg"), InputPort<double>("MinClearSec"),
			InputPort<double>("MinEnemyLOSDeg"), InputPort<double>("MinClearEnemyLOSDeg"),
			InputPort<double>("MaxEnemyLOSDeg"),
			InputPort<double>("MinLosDeg"), InputPort<double>("MaxLosDeg"),
			InputPort<double>("MinEnemyLosDeg"), InputPort<double>("MaxEnemyLosDeg"),
			InputPort<double>("MaxAbsClosureFtps"), InputPort<double>("NearPassRangeFt"),
			InputPort<double>("AbeamRangeFt"),
			InputPort<double>("MinRangeFt"), InputPort<double>("MaxRangeFt"),
			InputPort<double>("MinClosureFtps"), InputPort<double>("MaxClosureFtps"), InputPort<double>("ClosureBrakeRangeFt"),
			InputPort<double>("CueEnergyMarginM"),
			InputPort<double>("MaxSec"), InputPort<double>("OneMaxSec"), InputPort<double>("TwoMaxSec"),
			InputPort<double>("BreakJinkMaxSec"), InputPort<double>("BreakJinkEnemyLOSDeg"),
			InputPort<double>("ScissorsMaxSec"), InputPort<double>("ScissorsCommitSec"),
			InputPort<double>("PureLureMinSec"), InputPort<double>("PureLureMinRangeFt"),
			InputPort<double>("PureLureMaxRangeFt"), InputPort<double>("PureLureMaxAbsClosureFtps"),
			InputPort<double>("PureLureMinMyLOSDeg"), InputPort<double>("PureLureGuardDeg"),
			InputPort<double>("PureLureMaxEnemyLOSDeg"),
			InputPort<double>("NoCueSec"),
			InputPort<double>("CommitSec"), InputPort<double>("OneRearEnemyLOSDeg"), InputPort<double>("OneRearMaxRangeFt"),
			InputPort<double>("EnterKCAS"), InputPort<double>("MinKCAS"), InputPort<double>("MaxKCAS"),
			InputPort<double>("MaxAllowedEnemyLOSDeg"),
			InputPort<double>("OffensiveCommitSec"), InputPort<double>("LeadPredictionSec"),
			InputPort<double>("MaxLeadDeg"), InputPort<double>("MinAbsLosRateDegps"),
			InputPort<double>("MaxAbsLosRateDegps"), InputPort<double>("MinPhase"),
			InputPort<double>("MaxRangeBufferFt"),
			InputPort<double>("MaxEnemyLosRateDegps"), InputPort<double>("FriendlyLOSDeg"),
			InputPort<double>("EnemyBetterMarginDeg"), InputPort<double>("DamageMargin"),
			InputPort<double>("PotentialMargin"), InputPort<double>("MinNeutralSec"),
			InputPort<double>("MinThreatClearSec"), InputPort<double>("ContestGuardDeg"),
			InputPort<double>("ShotCommitSec"), InputPort<double>("ShotCommitMaxLosDeg"),
			InputPort<double>("EnemyDamageMargin"), InputPort<double>("MinAngularAdvDeg"),
			InputPort<double>("CounterMaxMyLOSDeg"), InputPort<double>("CounterMinEnemyLOSDeg"),
			InputPort<double>("CounterMinAngularAdvDeg"), InputPort<double>("CounterMaxRangeFt"),
			InputPort<double>("Node35Sec"), InputPort<double>("RecoverySec"),
			InputPort<double>("CaptureRangeFt"), InputPort<double>("CaptureLosDeg"),
			InputPort<bool>("Approach")
		};
	}

	CPPBlackBoard* CompetitionNode::board() { return getInput<CPPBlackBoard*>("BB").value(); }

	double CompetitionNode::number(const char* name, double fallback)
	{
		Optional<double> value = getInput<double>(name);
		return value ? value.value() : fallback;
	}

	NodeStatus UpdateGeometry::tick()
	{
		CPPBlackBoard* bb = board();
		const double dt = std::max(bb->DeltaSecond, 0.001);
		bb->Distance = static_cast<float>(bb->MyLocation_Cartesian.distance(bb->TargetLocaion_Cartesian));
		if (!bb->Enemy.empty())
		{
			bb->TargetBodyVelocity = bb->Enemy.at(0).BodyVelocity;
			bb->TargetKCAS_KT = bb->Enemy.at(0).KCAS;
			bb->HasPerfectTargetState = bb->Enemy.at(0).HasExtendedState;
		}
		const bool hasHistory = bb->PreviousMyLocation.lengthSquared() > 0.0 &&
			bb->PreviousTargetLocation.lengthSquared() > 0.0;
		const bool hasPerfectState = bb->HasPerfectMyState && bb->HasPerfectTargetState;
		if (hasPerfectState)
		{
			// NavigationData supplies body-axis velocities directly. Transform
			// them into the same North-East-Up frame used by pursuit points.
			bb->MyVelocity = bb->MyForwardVector * bb->MyBodyVelocity.X +
				bb->MyRightVector * bb->MyBodyVelocity.Y -
				bb->MyUpVector * bb->MyBodyVelocity.Z;
			bb->TargetVelocity = bb->TargetForwardVector * bb->TargetBodyVelocity.X +
				bb->TargetRightVector * bb->TargetBodyVelocity.Y -
				bb->TargetUpVector * bb->TargetBodyVelocity.Z;
			if (hasHistory)
			{
				bb->MyAcceleration = (bb->MyVelocity - bb->PreviousMyVelocity) / dt;
				bb->TargetAcceleration = (bb->TargetVelocity - bb->PreviousTargetVelocity) / dt;
			}
			bb->AltSpeed = static_cast<float>(bb->MyVelocity.Z);
		}
		else if (hasHistory)
		{
			bb->MyVelocity = (bb->MyLocation_Cartesian - bb->PreviousMyLocation) / dt;
			bb->TargetVelocity = (bb->TargetLocaion_Cartesian - bb->PreviousTargetLocation) / dt;
			bb->MyAcceleration = (bb->MyVelocity - bb->PreviousMyVelocity) / dt;
			bb->TargetAcceleration = (bb->TargetVelocity - bb->PreviousTargetVelocity) / dt;
			bb->AltSpeed = static_cast<float>((bb->MyLocation_Cartesian.Z - bb->PreviousMyLocation.Z) / dt);
		}
		else
		{
			bb->MyVelocity = bb->MyForwardVector * bb->MySpeed_MS;
			bb->TargetVelocity = bb->TargetForwardVector * bb->TargetSpeed_MS;
		}

		BT_Geometry::Vector3 line = direction(bb->MyLocation_Cartesian, bb->TargetLocaion_Cartesian, bb->MyForwardVector);
		bb->Los_Degree = static_cast<float>(bb->MyForwardVector.angleBetween(line) * DEG);
		bb->Los_Degree_Target = static_cast<float>(bb->TargetForwardVector.angleBetween(-line) * DEG);
		bb->MyAngleOff_Degree = static_cast<float>(bb->MyForwardVector.angleBetween(bb->TargetForwardVector) * DEG);
		bb->MyAspectAngle_Degree = bb->Los_Degree_Target;
		bb->ClosureRate_MS = static_cast<float>((bb->TargetVelocity - bb->MyVelocity).dot(line));
		bb->TimeToMerge = bb->ClosureRate_MS < -1.0f ? static_cast<float>(bb->Distance / -bb->ClosureRate_MS) : 999.0f;
		const double rangeFt = bb->Distance / FT;

		// Constant-turn (circular-arc) prediction, spec 4.1. Linear extrapolation
		// of a hard-turning target walks off its circle tangentially - chasing
		// that point parks us in a stable orbit OUTSIDE the fight (observed:
		// 117 s LeadIntercept loop vs the pure-pursuit baseline). Uses last
		// tick's turn circle (UpdateEnemyTurnCircle runs after this node).
		const double predictTime = clampValue(bb->TimeToMerge, 0.5, 2.0);
		BT_Geometry::Vector3 turnRadial = bb->TargetLocaion_Cartesian - bb->TargetTurnCenter;
		BT_Geometry::Vector3 turnAxis = turnRadial.cross(bb->TargetVelocity);
		if (bb->TargetTurnCircleValid && turnAxis.length() > 1.0 && turnRadial.length() > 1.0)
		{
			turnAxis.normalize();
			const double turnRate = bb->TargetVelocity.length() / std::max(static_cast<double>(bb->TargetTurnRadius_M), 1.0);
			bb->PredictedTargetLocation = bb->TargetTurnCenter +
				rotateAboutAxis(turnRadial, turnAxis, turnRate * predictTime);
		}
		else
			bb->PredictedTargetLocation = bb->TargetLocaion_Cartesian + bb->TargetVelocity * predictTime;
		if (!bb->Enemy.empty()) bb->TargetHealth = bb->Enemy.at(0).Resv1;

		if (hasHistory)
		{
			bb->MyLosRate_DegSec = static_cast<float>((bb->Los_Degree - bb->PreviousMyLos_Degree) / dt);
			bb->EnemyLosRate_DegSec = static_cast<float>((bb->Los_Degree_Target - bb->PreviousEnemyLos_Degree) / dt);
		}
		bb->PreviousMyLos_Degree = bb->Los_Degree;
		bb->PreviousEnemyLos_Degree = bb->Los_Degree_Target;

		// Mode classification with hysteresis (spec section 5).
		// AA_T = 180 - enemyLOS (enemy tail aspect), AA_A = 180 - myLOS.
		// OFFENSIVE  enter: AA_T < 60 AND myLOS < 90   / exit: AA_T > 80 OR myLOS > 110
		// DEFENSIVE  enter: AA_A < 60 AND enemyLOS < 90 / exit: AA_A > 80 OR enemyLOS > 110
		// Defensive wins on simultaneous entry.
		bb->EnergyAdvantage_M = static_cast<float>(
			(bb->MyLocation_Cartesian.Z + bb->MySpeed_MS * bb->MySpeed_MS / (2.0 * G)) -
			(bb->TargetLocaion_Cartesian.Z + bb->TargetSpeed_MS * bb->TargetSpeed_MS / (2.0 * G)));
		// The paper bases superiority on the 3/9 line and relative energy.
		// Use a 10-deg geometric hysteresis around that line and a similarity
		// band calibrated from the paper's own 424/496-kt example (~-498 m).
		// VER05: the simulator does not expose an EM chart, and earlier builds
		// let a modest energy deficit (-1000 m is common in the traces) erase
		// clear rear-hemisphere OBFM geometry.  Keep energy as a severe-loss
		// veto only; the cone contest rewards turning a positional advantage into
		// nose authority rather than abandoning it to HABFM/PureFallback.
		const bool energyNotSeverelyLosing = bb->EnergyAdvantage_M > -1800.0f;
		const bool energyNotWinning = bb->EnergyAdvantage_M < 900.0f;
		// VER09: superiority must express an actual angular advantage, not just
		// both fighters being near the 3/9 line.  The public starts are roughly
		// abeam (myLOS ~= enemyLOS ~= 90-110 deg); VER08 classified those
		// symmetric states OBFM and skipped the paper HABFM cue sequence.
		const double angularAdvantageDeg =
			static_cast<double>(bb->Los_Degree_Target - bb->Los_Degree);
		const int tacticalPhase = bb->RunningTime < 100.0 ? 1 : (bb->RunningTime < 150.0 ? 2 : 3);
		const double tacticalConeDeg = phaseConeDeg(tacticalPhase);
		const double tacticalMaxRangeFt = phaseMaxRangeFt(tacticalPhase);
		const bool actualEnemyWez =
			damageRate(bb->Los_Degree_Target, rangeFt, tacticalPhase) > 0.0;
		const bool predictedEnemyWez = predictedThreatCue(
			bb, rangeFt, tacticalConeDeg, tacticalMaxRangeFt,
			8.0, 900.0, 22.0, 60.0, 8.0, 1.5, 0.04, 15.0, 3000.0);
		const bool actualOwnWez =
			damageRate(bb->Los_Degree, rangeFt, tacticalPhase) > 0.0;
		bb->EnemyPursuitType = classifyEnemyPursuit(bb, rangeFt, tacticalPhase);
		const bool defensiveRecoveryActive =
			bb->RunningTime < bb->DefensiveRecoveryUntil && !actualEnemyWez;
		const bool priorOffensiveMode = bb->BFM == OBFM;
		const bool employCommitActive =
			priorOffensiveMode &&
			bb->RunningTime < bb->ShotCommitUntil &&
			rangeFt > 500.0 &&
			rangeFt < tacticalMaxRangeFt + 2200.0 &&
			bb->Los_Degree < std::max(18.0, tacticalConeDeg + 8.0) &&
			!actualEnemyWez;
		const bool offensiveEnter =
			rangeFt < 9000.0 &&
			bb->Los_Degree < 85.0f && bb->Los_Degree_Target > 95.0f &&
			angularAdvantageDeg > 20.0 && energyNotSeverelyLosing;
		const bool strictOffensiveStay =
			bb->Los_Degree < 100.0f && bb->Los_Degree_Target > 85.0f &&
			angularAdvantageDeg > 10.0 && energyNotSeverelyLosing;
		// Once the paper OBFM flow has identified a clear angular advantage,
		// keep the control-zone pursuit alive long enough to convert it.  VER04
		// dropped back to HABFM after FarBehind/FollowingHostile as soon as the
		// target's LOS crossed the strict 85-deg stay gate, even while ownship
		// still held a large relative angular advantage and no enemy WEZ existed.
		const bool captureOffensiveStay =
			bb->BFM == OBFM &&
			rangeFt < 14000.0 &&
			bb->Los_Degree < 88.0f &&
			bb->Los_Degree_Target > 45.0f &&
			angularAdvantageDeg > 24.0 &&
			energyNotSeverelyLosing;
		const bool offensiveStay = strictOffensiveStay || captureOffensiveStay;
		const bool defensiveEnter =
			rangeFt < 8500.0 &&
			bb->Los_Degree_Target < 30.0f && bb->Los_Degree > 105.0f &&
			angularAdvantageDeg < -45.0 && energyNotWinning;
		const bool defensiveStay =
			rangeFt < 9000.0 &&
			bb->ThreatClearTime < 2.0f &&
			bb->Los_Degree_Target < 42.0f && bb->Los_Degree > 95.0f &&
			angularAdvantageDeg < -30.0 && energyNotWinning;
		const double defensiveClosingFtpsForMode =
			std::max(0.0, -static_cast<double>(bb->ClosureRate_MS) / FT);
		const bool lowClosurePureRecoveryGrace =
			bb->EnemyPursuitType == EPT_PURE &&
			bb->DefensiveRecoveryUntil > 0.0f &&
			rangeFt > 1300.0 &&
			rangeFt < 6500.0 &&
			defensiveClosingFtpsForMode < 160.0 &&
			bb->Los_Degree_Target > tacticalConeDeg + 3.5f &&
			bb->Los_Degree > 100.0f &&
			!actualEnemyWez &&
			!actualOwnWez;
		const bool habfmBridgeActive = bb->HABFMNextManeuverTask >= 0 &&
			bb->RunningTime < bb->HABFMPullToHUDUntil;
		const int previousBfm = bb->BFM;
		if (bb->RunningTime < bb->DefensiveCommitUntil ||
			actualEnemyWez ||
			(!defensiveRecoveryActive &&
				!lowClosurePureRecoveryGrace &&
				(predictedEnemyWez || defensiveEnter || (bb->BFM == DBFM && defensiveStay)))) bb->BFM = DBFM;
		else if (habfmBridgeActive && bb->RunningTime >= bb->ShotCommitUntil && !offensiveEnter &&
			!actualOwnWez && !employCommitActive)
			bb->BFM = HABFM;
		else if ((priorOffensiveMode &&
				(bb->RunningTime < bb->OffensiveCommitUntil || bb->RunningTime < bb->ShotCommitUntil)) ||
			actualOwnWez || employCommitActive ||
			offensiveEnter || (priorOffensiveMode && offensiveStay)) bb->BFM = OBFM;
		else bb->BFM = HABFM;
		if (bb->BFM == OBFM && (offensiveEnter || offensiveStay))
			bb->OffensiveCommitUntil = static_cast<float>(std::max(static_cast<double>(bb->OffensiveCommitUntil),
				bb->RunningTime + 1.2));
		updateControlZoneState(bb, rangeFt, actualEnemyWez);
		if (bb->BFM != HABFM)
		{
			bb->PendingScissors = false;
			bb->HABFMNextManeuverTask = -1;
			bb->HABFMPullToHUDUntil = 0.0f;
		}
		bb->NeutralTime = bb->BFM == HABFM ? bb->NeutralTime + static_cast<float>(dt) : 0.0f;
		const double defensiveClosingFtps =
			std::max(0.0, -static_cast<double>(bb->ClosureRate_MS) / FT);
		const bool contestThreatClearGeometry =
			bb->BFM == DBFM &&
			previousBfm == DBFM &&
			!actualEnemyWez &&
			bb->Los_Degree_Target > tacticalConeDeg + 5.0f &&
			(defensiveClosingFtps < 210.0 ||
			 bb->ClosureRate_MS / FT > 0.0 ||
			 rangeFt > tacticalMaxRangeFt + 800.0);
		const bool dbfmThreatClearGeometry =
			contestThreatClearGeometry ||
			(bb->BFM == DBFM &&
			 previousBfm == DBFM &&
			 bb->Los_Degree_Target > 25.0f &&
			 angularAdvantageDeg > -30.0);
		bb->ThreatClearTime = dbfmThreatClearGeometry
			? bb->ThreatClearTime + static_cast<float>(dt)
			: 0.0f;
		const BT_Geometry::Vector3 currentLevelForward = levelForward(bb);
		if (bb->PreviousLevelForward.length() > 0.5)
		{
			const double signedTurn = signedPlanarAngleDeg(
				bb->PreviousLevelForward, currentLevelForward) / dt;
			bb->PlanarTurnRateSigned_DegSec = static_cast<float>(signedTurn);
			bb->PlanarTurnRate_DegSec = static_cast<float>(std::abs(signedTurn));
		}
		else
		{
			bb->PlanarTurnRateSigned_DegSec = 0.0f;
			bb->PlanarTurnRate_DegSec = 0.0f;
		}
		bb->PreviousLevelForward = currentLevelForward;

		BT_Geometry::Vector3 currentTargetLevel = bb->TargetVelocity.length() > 1.0
			? bb->TargetVelocity : bb->TargetForwardVector;
		currentTargetLevel.Z = 0.0;
		if (currentTargetLevel.length() < 0.01)
			currentTargetLevel = BT_Geometry::Vector3(1, 0, 0);
		else
			currentTargetLevel.normalize();
		if (bb->PreviousTargetLevelForward.length() > 0.5)
			bb->TargetPlanarTurnRateSigned_DegSec = static_cast<float>(
				signedPlanarAngleDeg(bb->PreviousTargetLevelForward, currentTargetLevel) / dt);
		else
			bb->TargetPlanarTurnRateSigned_DegSec = 0.0f;
		bb->PreviousTargetLevelForward = currentTargetLevel;
		if (std::abs(bb->PlanarTurnRateSigned_DegSec) > 1.0f &&
			std::abs(bb->TargetPlanarTurnRateSigned_DegSec) > 1.0f)
			bb->CircleDirectionRelation =
				bb->PlanarTurnRateSigned_DegSec * bb->TargetPlanarTurnRateSigned_DegSec >= 0.0f ? 1 : -1;
		else
			bb->CircleDirectionRelation = 0;

		if (bb->LockedManeuverTask >= 0)
		{
			if (bb->PreviousManeuverForward.length() > 0.5)
			{
				const double signedDelta = signedPlanarAngleDeg(
					bb->PreviousManeuverForward, currentLevelForward);
				const double commandedProgress = signedDelta * bb->LockedManeuverSide;
				bb->ManeuverTurnDegrees = static_cast<float>(std::max(0.0,
					static_cast<double>(bb->ManeuverTurnDegrees) + commandedProgress));
			}
			bb->PreviousManeuverForward = currentLevelForward;
		}
		bb->PreviousMyLocation = bb->MyLocation_Cartesian;
		bb->PreviousTargetLocation = bb->TargetLocaion_Cartesian;
		bb->PreviousMyVelocity = bb->MyVelocity;
		bb->PreviousTargetVelocity = bb->TargetVelocity;
		return NodeStatus::SUCCESS;
	}

	NodeStatus UpdateEnergy::tick()
	{
		CPPBlackBoard* bb = board();
		bb->EnergyAdvantage_M = static_cast<float>((bb->MyLocation_Cartesian.Z + bb->MySpeed_MS * bb->MySpeed_MS / (2.0 * G)) -
			(bb->TargetLocaion_Cartesian.Z + bb->TargetSpeed_MS * bb->TargetSpeed_MS / (2.0 * G)));
		return NodeStatus::SUCCESS;
	}

	NodeStatus UpdateEnemyTurnCircle::tick()
	{
		CPPBlackBoard* bb = board();
		const double speed = bb->TargetVelocity.length();
		BT_Geometry::Vector3 tangent = normalized(bb->TargetVelocity);
		BT_Geometry::Vector3 normalAcceleration = bb->TargetAcceleration -
			tangent * bb->TargetAcceleration.dot(tangent);
		const double normalAccelerationMagnitude = normalAcceleration.length();
		const double measuredTurnRateDegSec = speed > 1.0
			? (normalAccelerationMagnitude / speed) * DEG
			: 0.0;
		// A shallow course correction produces a very large mathematical
		// circle that is not the combat turn circle used by BEM. Only retain
		// an established tactical turn with a reachable radius.
		const double measuredRadius = normalAccelerationMagnitude > 0.01
			? speed * speed / normalAccelerationMagnitude
			: 1.0e9;
		if (speed > 1.0 && measuredTurnRateDegSec > 6.0 && measuredRadius < 6000.0)
		{
			const double radius = clampValue(measuredRadius, 100.0, 6000.0);
			bb->TargetTurnRadius_M = static_cast<float>(radius);
			bb->TargetTurnRate_DegSec = static_cast<float>((speed / radius) * DEG);
			bb->TargetTurnCenter = bb->TargetLocaion_Cartesian + normalized(normalAcceleration) * radius;
			bb->TargetTurnCircleValid = true;
		}
		else
		{
			bb->TargetTurnRadius_M = 0.0f;
			bb->TargetTurnRate_DegSec = 0.0f;
			bb->TargetTurnCenter = bb->TargetLocaion_Cartesian;
			bb->TargetTurnCircleValid = false;
		}
		return NodeStatus::SUCCESS;
	}

	NodeStatus UpdatePhase::tick()
	{
		CPPBlackBoard* bb = board();
		bb->Phase = bb->RunningTime < 100.0 ? 1 : (bb->RunningTime < 150.0 ? 2 : 3);
		return NodeStatus::SUCCESS;
	}

	NodeStatus UpdateDamageScore::tick()
	{
		CPPBlackBoard* bb = board();
		const double rangeFt = bb->Distance / FT;
		bb->MyDamageRate = static_cast<float>(damageRate(bb->Los_Degree, rangeFt, bb->Phase));
		bb->EnemyDamageRate = static_cast<float>(damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase));
		bb->MyDamageBand = damageBand(bb->Los_Degree, rangeFt, bb->Phase);
		bb->EnemyDamageBand = damageBand(bb->Los_Degree_Target, rangeFt, bb->Phase);
		bb->TargetInMyCone = bb->MyDamageRate > 0.0f;
		bb->OwnshipInEnemyCone = bb->EnemyDamageRate > 0.0f;
		bb->EstimatedDamageDealt += static_cast<float>(bb->MyDamageRate * bb->DeltaSecond);
		bb->EstimatedDamageTaken += static_cast<float>(bb->EnemyDamageRate * bb->DeltaSecond);
		bb->DamageDifference = bb->EstimatedDamageDealt - bb->EstimatedDamageTaken;
		return NodeStatus::SUCCESS;
	}

	NodeStatus BaselineCorePolicy::tick()
	{
		CPPBlackBoard* bb = board();
		bb->ActiveCompetitionTask = -2;
		const double corner = 400.0 * KNOT;

		// Ground avoidance: a fixed 2000 ft trigger let a steep, fast dive blow
		// through the 1000 ft hard deck before it could pull out. Use the same
		// sink-rate / pull-up-radius aware recovery as the candidate so the
		// baseline is a fair sparring partner rather than crashing on its own.
		if (bb->MyLocation_Cartesian.Z / FT < groundRecoveryAltitudeFt(bb, 1000.0, 500.0))
		{
			bb->VP_Cartesian = groundRecoveryVP(bb);
			bb->VPVelocity = BT_Geometry::Vector3(0, 0, 0);
			bb->VPVelocityValid = false;
			bb->Throttle = 1.0f;
			return NodeStatus::SUCCESS;
		}

		if (bb->BFM == DBFM)
		{
			const double side = turnSide(bb);
			bb->VP_Cartesian = bb->MyLocation_Cartesian + bb->MyRightVector * (side * 6000.0) +
				bb->MyForwardVector * 500.0;
			bb->VPVelocity = BT_Geometry::Vector3(0, 0, 0);
			bb->VPVelocityValid = false;
		}
		else
		{
			// The frozen baseline deliberately uses pure pursuit only.
			bb->VP_Cartesian = bb->TargetLocaion_Cartesian;
			bb->VPVelocity = bb->TargetVelocity;
			bb->VPVelocityValid = true;
		}

		bb->Throttle = bb->MySpeed_MS < corner ? 1.0f : 0.65f;
		return NodeStatus::SUCCESS;
	}

	CompetitionCondition::CompetitionCondition(const std::string& name, const NodeConfiguration& config,
		CompetitionConditionKind kind) : CompetitionNode(name, config), kind_(kind) {}

	NodeStatus CompetitionCondition::tick()
	{
		CPPBlackBoard* bb = board();
		bool result = false;
		const double rangeFt = bb->Distance / FT;
		const double maxRangeFt = phaseMaxRangeFt(bb->Phase);
		const double coneDeg = phaseConeDeg(bb->Phase);
		switch (kind_)
		{
		case COND_GROUND_DANGER:
			result = bb->MyLocation_Cartesian.Z / FT <
				groundRecoveryAltitudeFt(bb, number("FloorFt", 1000.0), number("MarginFt", 500.0));
			break;
		case COND_DEFENSIVE:
		{
			// VER05: do not let the coarse BFM==DBFM classifier alone select the
			// HardTurn fallback.  In VER04 draws, many ticks were geometry-DBFM
			// without an actual cone threat, which interrupted OBFM/HABFM scoring.
			// UnderFire and narrow PredictedThreat now live inside DBFM, so this
			// gate must recognize those cues rather than relying on a top-level
			// interrupt to set DefensiveCommitUntil first.
			const bool enemyActualGun = damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) > 0.0;
			const bool ownActualGun = damageRate(bb->Los_Degree, rangeFt, bb->Phase) > 0.0;
			const bool enemyOwnsBetterMutualShot =
				ownActualGun &&
				bb->Los_Degree_Target + number("MutualWezEnemyMarginDeg", 1.0) < bb->Los_Degree;
			const bool actualGunThreat =
				enemyActualGun && (!ownActualGun || enemyOwnsBetterMutualShot);
			const bool predictedGunThreat = predictedThreatCue(
				bb, rangeFt, coneDeg, maxRangeFt,
				number("PredictedEnemyLOSDeg", 8.0),
				number("PredictedRangeBufFt", 900.0),
				number("PredictedFriendlyLOSDeg", 22.0),
				number("MinMyLOSDeg", 60.0),
				number("PredictedMaxEnemyLosRateDegps", 8.0),
				number("PredictedEnemyBetterMarginDeg", 1.5),
				number("PredictedPotentialMargin", 0.04),
				number("PredictedNeutralSec", 15.0),
				number("PredictedStalemateRangeFt", 3000.0));
			if (actualGunThreat)
			{
				bb->LockedManeuverTask = -1;
				bb->DefensiveCommitUntil = static_cast<float>(
					std::max(static_cast<double>(bb->DefensiveCommitUntil),
						bb->RunningTime + number("CommitSec", 0.6)));
			}
			const double angularAdvantage =
				static_cast<double>(bb->Los_Degree_Target - bb->Los_Degree);
			const double closingFtps =
				std::max(0.0, -static_cast<double>(bb->ClosureRate_MS) / FT);
			const bool defensiveRecoveryActive =
				bb->RunningTime < bb->DefensiveRecoveryUntil && !enemyActualGun;
			const bool urgentPredictedDuringRecovery =
				defensiveRecoveryActive &&
				predictedGunThreat &&
				bb->Los_Degree_Target < coneDeg + 2.0f &&
				rangeFt < maxRangeFt + 600.0 &&
				bb->Los_Degree > 100.0f &&
				angularAdvantage < -50.0;
			const bool establishedDefensiveGeometry = bb->BFM == DBFM &&
				rangeFt < 7500.0 && bb->ThreatClearTime < 2.0f &&
				bb->Los_Degree_Target < 35.0f &&
				bb->Los_Degree > 105.0f && angularAdvantage < -55.0;
			const bool dbfmNeedsResolution = bb->BFM == DBFM &&
				rangeFt < 6500.0 &&
				bb->Los_Degree_Target < 50.0f &&
				bb->Los_Degree > 90.0f &&
				angularAdvantage < -35.0;
			const bool purePursuitDefensiveThreat =
				bb->EnemyPursuitType == EPT_PURE &&
				rangeFt < 6500.0 &&
				bb->Los_Degree_Target < 18.0f &&
				bb->Los_Degree > 85.0f &&
				angularAdvantage < -45.0 &&
				closingFtps > 420.0 &&
				!ownActualGun;
			const bool fastRearPreConeThreat =
				rangeFt < maxRangeFt + number("PredictedRangeBufFt", 900.0) &&
				bb->Los_Degree_Target < coneDeg + number("ContestGuardDeg", 18.0) &&
				bb->Los_Degree > number("MinMyLOSDeg", 60.0) &&
				angularAdvantage < -45.0 &&
				closingFtps > 350.0 &&
				bb->EnemyLosRate_DegSec < -6.0f &&
				!ownActualGun &&
				!isFastHeadOnMerge(bb, rangeFt) &&
				!isNeutralHeadOnRemerge(bb, rangeFt);
			const bool rearLagDefensiveThreat =
				bb->EnemyPursuitType == EPT_LAG &&
				rangeFt < number("LagThreatMaxRangeFt", 6500.0) &&
				bb->Los_Degree > number("LagThreatMinMyLOSDeg", 125.0) &&
				bb->Los_Degree_Target < number("LagThreatMaxEnemyLOSDeg", 52.0) &&
				angularAdvantage < number("LagThreatMaxAngularAdvDeg", -65.0) &&
				bb->EnemyLosRate_DegSec < number("LagThreatMaxEnemyLosRateDegps", -4.0) &&
				!ownActualGun;
			const bool lowClosurePureRecoveryGrace =
				bb->EnemyPursuitType == EPT_PURE &&
				bb->DefensiveRecoveryUntil > 0.0f &&
				rangeFt > 1300.0 &&
				rangeFt < 6500.0 &&
				closingFtps < 160.0 &&
				bb->Los_Degree_Target > coneDeg + 3.5f &&
				bb->Los_Degree > 100.0f &&
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0 &&
				!ownActualGun;
			const double scissorsElapsed = bb->RunningTime - bb->ManeuverStartTime;
			const bool habfmScissorsCommitActive =
				bb->BFM == HABFM &&
				bb->LockedManeuverTask == static_cast<int>(TASK_SCISSORS) &&
				scissorsElapsed >= 0.0 &&
				scissorsElapsed < number("ScissorsCommitSec", 0.85) &&
				rangeFt > number("MinRangeFt", 1200.0) &&
				bb->Los_Degree_Target > std::max(coneDeg + number("ContestGuardDeg", 18.0), 22.0) &&
				!enemyActualGun;
			result = actualGunThreat ||
				(!defensiveRecoveryActive &&
					!lowClosurePureRecoveryGrace &&
					!habfmScissorsCommitActive &&
					(predictedGunThreat ||
					 bb->RunningTime < bb->DefensiveCommitUntil ||
					 establishedDefensiveGeometry || dbfmNeedsResolution ||
					 purePursuitDefensiveThreat || fastRearPreConeThreat ||
					 rearLagDefensiveThreat)) ||
				urgentPredictedDuringRecovery;
			if (fastRearPreConeThreat && !habfmScissorsCommitActive)
			{
				bb->LockedManeuverTask = -1;
				bb->DefensiveCommitUntil = static_cast<float>(std::max(
					static_cast<double>(bb->DefensiveCommitUntil),
					bb->RunningTime + number("CommitSec", 0.6)));
			}
			if (rearLagDefensiveThreat && !habfmScissorsCommitActive)
			{
				bb->LockedManeuverTask = -1;
				bb->DefensiveCommitUntil = static_cast<float>(std::max(
					static_cast<double>(bb->DefensiveCommitUntil),
					bb->RunningTime + number("LagThreatCommitSec", 0.8)));
			}
			if (result)
				bb->BFM = DBFM;
			break;
		}

		case COND_UNDER_FIRE:
		{
			const double threatRangeBufFt = number("RangeBufFt", 0.0);
			const bool blindRearThreat =
				bb->Los_Degree >= number("MinMyLOSDeg", 0.0);
			const bool enemyStrictCone =
				bb->Los_Degree_Target < coneDeg && rangeFt > 500.0 && rangeFt < maxRangeFt;
			const bool enemyNearCone =
				blindRearThreat &&
				bb->Los_Degree_Target < std::max(coneDeg, number("GuardDeg", coneDeg + 1.0)) &&
				rangeFt > 500.0 && rangeFt < maxRangeFt + threatRangeBufFt;
			const double closingFtps = std::max(0.0, -static_cast<double>(bb->ClosureRate_MS) / FT);
			const bool enemyClosingEnough =
				closingFtps >= number("MinClosureFtps", 0.0);
			const bool ownNearCone =
				bb->Los_Degree < number("FriendlyLOSDeg", coneDeg + 3.0) &&
				rangeFt > 500.0 && rangeFt < maxRangeFt + number("RangeBufFt", 400.0);
			const bool enemyConeClosingFast =
				bb->EnemyLosRate_DegSec < number("MaxEnemyLosRateDegps", -4.0);
			const bool scissorsRecoveryActive =
				bb->LockedManeuverTask == static_cast<int>(TASK_SCISSORS) &&
				!enemyStrictCone;
			result = enemyStrictCone ||
				(enemyNearCone && enemyConeClosingFast && enemyClosingEnough &&
				 !ownNearCone && !scissorsRecoveryActive);
			if (result)
			{
				bb->LockedManeuverTask = -1;
				bb->DefensiveCommitUntil = static_cast<float>(
					std::max(static_cast<double>(bb->DefensiveCommitUntil),
						bb->RunningTime + number("CommitSec", 3.0)));
				Optional<bool> breakJinkLock = getInput<bool>("BreakJinkLock");
				if (breakJinkLock && breakJinkLock.value())
					startManeuverLock(bb, static_cast<int>(TASK_BREAK_JINK), turnSide(bb));
			}
			break;
		}
		case COND_PREDICTED_THREAT:
		{
			// Phase-aware predicted threat. A pure-pursuit baseline often points
			// near our aircraft, so do not break just because enemyLOS is small.
			// Break only when the enemy's near-cone potential clearly dominates
			// our own near-cone potential and no shot commit is active.
			const double gateDeg = clampValue(number("EnemyLOSDeg", coneDeg + number("GuardDeg", 3.0)),
				coneDeg + 1.0, coneDeg + 8.0);
			// VER06: do not let a broad shot commit mask a clearly better enemy
			// near-cone. VER05 losses show enemyLOS < 1 deg appearing only after
			// Track/commit had held too long with myLOS still 50-150 deg.
			result = predictedThreatCue(
				bb, rangeFt, coneDeg, maxRangeFt,
				gateDeg,
				number("RangeBufFt", 400.0),
				number("FriendlyLOSDeg", gateDeg + 4.0),
				number("MinMyLOSDeg", 0.0),
				number("MaxEnemyLosRateDegps", 0.6),
				number("EnemyBetterMarginDeg", 1.25),
				number("PotentialMargin", 0.10),
				number("NeutralSec", 15.0),
				number("StalemateRangeFt", 3000.0));
			if (result)
			{
				bb->LockedManeuverTask = -1;
				bb->DefensiveCommitUntil = static_cast<float>(
					std::max(static_cast<double>(bb->DefensiveCommitUntil),
						bb->RunningTime + number("CommitSec", 0.45)));
			}
			break;
		}

		case COND_OFFENSIVE:
		{
			// OBFM owns the paper Employ/Gun and shot-commit hysteresis. The
			// top-level no longer has an ActualGunWEZ shortcut, so this gate must
			// open the offensive subtree when the WEZ cue itself is present.
			const bool ownActualGun = damageRate(bb->Los_Degree, rangeFt, bb->Phase) > 0.0;
			const bool enemyActualGun = damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) > 0.0;
			const bool activeEmployCommit =
				bb->BFM == OBFM &&
				bb->RunningTime < bb->ShotCommitUntil &&
				rangeFt > 500.0 &&
				rangeFt < maxRangeFt + 2200.0 &&
				bb->Los_Degree < std::max(18.0, coneDeg + 8.0) &&
				!enemyActualGun;
			result = bb->BFM == OBFM || ownActualGun || activeEmployCommit;
			break;
		}
		case COND_IN_MY_WEZ:
		{
			// BEM block 7 asks whether the hostile is in the actual WEZ.
			// Keep this gate literal: the hostile must be inside the phase damage
			// cone. Near-but-not-scoring states stay in the lead/control-zone
			// conversion path instead of latching pure Track at 1-3 deg and
			// high closure.
			const double enterDeg = number("EnterDeg", coneDeg);
			const bool candidate = bb->Los_Degree < enterDeg &&
				rangeFt > 500.0 &&
				rangeFt < maxRangeFt + number("RangeBufFt", 0.0);
			// Do not let Track hysteresis mask DBFM while the adversary is
			// converging on a better near-WEZ solution. Waiting until it has
			// crossed the strict damage cone makes the defensive response too
			// late in a high-closure merge.
			const bool hostileOwnsBetterNearWez = enemyHasBetterNearCone(
				bb, rangeFt, coneDeg, maxRangeFt, 5.0, 1.5);
			result = candidate &&
				!hostileOwnsBetterNearWez &&
				!weakOpeningEdgeShot(bb, rangeFt, maxRangeFt, coneDeg);
			if (result)
				bb->ShotCommitUntil = static_cast<float>(std::max(static_cast<double>(bb->ShotCommitUntil),
					bb->RunningTime + number("ShotCommitSec", 1.2)));
			break;
		}
		case COND_OVERSHOOT:
			if (bb->BFM == DBFM)
			{
				// DBFM supplemental flow: when the attacker has passed close
				// enough that range is opening and its nose is no longer in a
				// gun solution, transition from the initial break/hard turn to
				// scissors instead of unloading into another neutral re-entry.
				const bool enemyActualCone =
					damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) > 0.0;
				const double closingFtps = std::max(0.0,
					-static_cast<double>(bb->ClosureRate_MS) / FT);
				const bool emergencyConeEscape =
					enemyActualCone &&
					(rangeFt < number("EmergencyRangeFt", 2600.0) ||
					 closingFtps > number("EmergencyClosureFtps", 450.0));
				const bool classicalOvershoot =
					rangeFt > number("MinRangeFt", 700.0) &&
					rangeFt < number("RangeFt", 3000.0) &&
					bb->ClosureRate_MS / FT > number("OpeningFtps", 100.0) &&
					bb->Los_Degree > number("MinLosDeg", 90.0) &&
					bb->Los_Degree_Target > number("MinEnemyLOSDeg", 55.0) &&
					!emergencyConeEscape;
				const double openingFtps = std::max(0.0,
					static_cast<double>(bb->ClosureRate_MS) / FT);
				const bool forcedFlatScissors =
					bb->ThreatClearTime > std::max(number("MinSec", 1.2), 1.8) &&
					rangeFt > number("MinRangeFt", 1100.0) &&
					rangeFt < number("MaxRangeFt", 4200.0) &&
					openingFtps > number("OpeningFtps", 100.0) &&
					std::abs(bb->ClosureRate_MS / FT) < number("MaxAbsClosureFtps", 260.0) &&
					bb->Los_Degree > number("MinMyLOSDeg", 105.0) &&
					bb->Los_Degree_Target > number("MinEnemyLOSDeg", 55.0) &&
					bb->Los_Degree_Target < number("EnemyLOSDeg", 85.0) &&
					!enemyActualCone &&
					!emergencyConeEscape;
				const bool pureLureScissors =
					bb->EnemyPursuitType == EPT_PURE &&
					bb->ThreatClearTime > std::max(number("MinSec", 1.2), number("PureLureMinSec", 4.0)) &&
					rangeFt > number("PureLureMinRangeFt", 2600.0) &&
					rangeFt < number("PureLureMaxRangeFt", 4300.0) &&
					std::abs(bb->ClosureRate_MS / FT) < number("PureLureMaxAbsClosureFtps", 160.0) &&
					bb->Los_Degree > number("PureLureMinMyLOSDeg", 150.0) &&
					bb->Los_Degree_Target > phaseConeDeg(bb->Phase) + number("PureLureGuardDeg", 3.0) &&
					bb->Los_Degree_Target < number("PureLureMaxEnemyLOSDeg", 12.0) &&
					!enemyActualCone &&
					!emergencyConeEscape;
				result = classicalOvershoot || forcedFlatScissors || pureLureScissors;
				if (result)
				{
					// Paper DBFM does not merely note the overshoot and then
					// fall immediately back to another neutral circle.  Arm a
					// short scissors lock so the reversal can actually develop
					// for a few ticks after the pure-threat break succeeds.
					if (bb->LockedManeuverTask != static_cast<int>(TASK_SCISSORS))
						startManeuverLock(bb, static_cast<int>(TASK_SCISSORS), turnSide(bb));
					bb->DefensiveCommitUntil = static_cast<float>(std::max(
						static_cast<double>(bb->DefensiveCommitUntil),
						bb->RunningTime + number("ScissorsCommitSec", 1.0)));
				}
			}
			else
				result = rangeFt < number("RangeFt", 2000.0) &&
					-bb->ClosureRate_MS / FT > number("ClosureFtps", 400.0) &&
					bb->Los_Degree > number("MinLosDeg", 0.0) &&
					bb->Los_Degree_Target > number("MinEnemyLOSDeg", 0.0) &&
					bb->MyAngleOff_Degree > number("AngleOffDeg", 45.0);
			break;
		// AA_T = 180 - enemyLOS. "Far behind" means we occupy the target's
		// rear hemisphere, not merely that our own nose points near it.
		case COND_FAR_BEHIND:
			result = rangeFt > number("RangeFt", 4000.0) &&
				bb->Los_Degree_Target > number("MinEnemyLOSDeg", 120.0);
			break;
		case COND_ENDGAME_DENY:
			// Score-denial extension is not part of Yang's BEM flow and is not
			// required by the simplified rules. Keep the legacy XML branch
			// disabled for paper fidelity.
			result = false;
			break;
		case COND_FAR_NEUTRAL:
		{
			const bool captured = rangeFt < number("CaptureRangeFt", 5500.0) &&
				bb->Los_Degree < number("CaptureLosDeg", 50.0);
			const bool committedRejoin = bb->RejoinTaskKind == 1 && !captured &&
				(bb->RunningTime < bb->RejoinCommitUntil ||
				 rangeFt > number("CaptureRangeFt", 5500.0) ||
				 bb->Los_Degree > number("CaptureLosDeg", 50.0));
			result = bb->BFM == HABFM &&
				(rangeFt > number("RangeFt", 6000.0) || committedRejoin);
			break;
		}
		case COND_PRE_MERGE:
		{
			// Use the BEM pre-merge block both for the initial abeam start and
			// for later neutral head-on re-merges. The latter was previously
			// falling through to PurePursuit/HardTurn and repeatedly producing
			// scoreless high-closure passes.
			const bool abeamEntry = bb->BFM == HABFM &&
				rangeFt < number("RangeFt", 5000.0) &&
				bb->Los_Degree > number("MinLosDeg", 60.0) &&
				bb->Los_Degree < number("MaxLosDeg", 120.0) &&
				bb->Los_Degree_Target > number("MinEnemyLosDeg", 60.0) &&
				bb->Los_Degree_Target < number("MaxEnemyLosDeg", 120.0) &&
				std::abs(bb->ClosureRate_MS / FT) < number("MaxAbsClosureFtps", 300.0);
			result = abeamEntry || isNeutralHeadOnRemerge(bb, rangeFt);
			break;
		}
		case COND_POST_MERGE:
			// Select one/two-circle at the merge, not as soon as HABFM is
			// classified. Turning at a 4000-6000 ft head-on separation exposed
			// our tail before the pass. Allow a short lead turn inside 1 s TTM
			// or open the gate after closure changes sign at the pass.
			// Yield to Scissors only for the close, low-energy stalemate
			// described by BEM block 21. A neutral winning-cue result by
			// itself continues HABFM and may start another one/two-circle.
			// Closure >= 0 alone is not a merge cue for an abeam start; it is
			// true at t=0 while both aircraft are still side-by-side. Require a
			// near pass, an imminent merge, or clear nose/tail geometry.
		{
			const bool abeamMergeStart =
				rangeFt <= number("AbeamRangeFt", 3500.0) &&
				bb->Los_Degree > 60.0f && bb->Los_Degree < 120.0f &&
				bb->Los_Degree_Target > 60.0f && bb->Los_Degree_Target < 120.0f;
			// The PreMerge branch is above this one in the selector.  Once its
			// entry-window cue releases, start the paper one/two-circle decision
			// rather than falling through to far PurePursuit merely because the
			// offset briefly drove both LOS angles beyond 120 degrees.
			const bool completedAbeamEntry =
				bb->NeutralTime > 0.45f &&
				rangeFt < number("NearPassRangeFt", 1500.0) + 800.0;
			const double closingFtps = std::max(0.0, -static_cast<double>(bb->ClosureRate_MS) / FT);
			const bool closeTerminalConversion =
				bb->Phase <= 1 &&
				rangeFt > 900.0 &&
				rangeFt < number("TerminalHoldRangeFt", 3600.0) &&
				bb->Los_Degree > 14.0f &&
				bb->Los_Degree < 58.0f &&
				bb->Los_Degree_Target > phaseConeDeg(bb->Phase) + 5.0 &&
				closingFtps > 450.0 &&
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0;
			const bool closeNoseThreatRecontest =
				rangeFt > number("RecontestMinRangeFt", 1800.0) &&
				rangeFt < number("RecontestMaxRangeFt", 4400.0) &&
				bb->Los_Degree > number("RecontestMinMyLOSDeg", 115.0) &&
				bb->Los_Degree_Target < number("RecontestMaxEnemyLOSDeg", 18.0) &&
				std::abs(bb->ClosureRate_MS / FT) < number("RecontestMaxAbsClosureFtps", 260.0) &&
				bb->DamageDifference <= number("RecontestMaxDamageAdvantage", 0.05) &&
				damageRate(bb->Los_Degree, rangeFt, bb->Phase) <= 0.0 &&
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0;
			result = bb->BFM == HABFM &&
				bb->RunningTime >= bb->ManeuverCooldownUntil &&
				bb->RunningTime >= bb->HABFMPullToHUDUntil &&
				bb->LockedManeuverTask < 0 &&
				(abeamMergeStart || completedAbeamEntry ||
				 rangeFt < number("NearPassRangeFt", 1500.0) ||
				 closeNoseThreatRecontest ||
				 bb->TimeToMerge <= 1.0f ||
				 (bb->ClosureRate_MS >= 0.0f &&
				  (bb->Los_Degree < 60.0f || bb->Los_Degree_Target < 60.0f))) &&
				!closeTerminalConversion &&
				!isCloseLowEnergyStalemate(bb, rangeFt, number("NeutralSec", 15.0));
			break;
		}
		// Legacy energy decorator retained for older XMLs. The active
		// competition XML uses COND_ROOM_TO_MANEUVER below for the HABFM
		// one/two-circle split.
		case COND_ENERGY_ADVANTAGE:
		{
			// Older trees used this as a broad high-energy/room predicate.
			// Keep it backward-compatible; do not use it for the current
			// competition HABFM split.
			result = rangeFt > 3000.0 || hasHighManeuverEnergy(bb);
			break;
		}
		case COND_STALEMATE:
		{
			// Paper block 21 is reached when HABFM one/two-circle has no
			// winning cue.  Keep the old low-energy stalemate gate, but also
			// consume PendingScissors after a short neutral dwell, unless either
			// fighter already has an immediate cone opportunity.
			const bool immediateOwnCone = damageRate(bb->Los_Degree, rangeFt, bb->Phase) > 0.0;
			const bool immediateEnemyCone = damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) > 0.0;
			Optional<bool> closeNoCue = getInput<bool>("CloseNoCue");
			const bool genericStalemateNode = !(closeNoCue && closeNoCue.value());
			const bool lowEnergyScissorsGeometry =
				genericStalemateNode &&
				std::abs(bb->ClosureRate_MS / FT) < number("MaxAbsClosureFtps", 500.0) &&
				bb->Los_Degree > number("MinLosDeg", 0.0) &&
				bb->Los_Degree < number("MaxLosDeg", 180.0) &&
				bb->Los_Degree_Target > number("MinEnemyLOSDeg", 0.0) &&
				bb->Los_Degree_Target < number("MaxEnemyLOSDeg", 180.0);
			const bool lowEnergyStalemate = isCloseLowEnergyStalemate(
				bb, rangeFt, number("NeutralSec", 15.0)) && lowEnergyScissorsGeometry;
			const bool pendingNoCue = bb->PendingScissors &&
				isCloseLowEnergyStalemate(bb, rangeFt, number("NeutralSec", 15.0)) &&
				rangeFt < number("RangeFt", 3000.0) &&
				!immediateOwnCone && !immediateEnemyCone;
			const bool closeNoWinningCue =
				closeNoCue && closeNoCue.value() &&
				rangeFt > number("MinRangeFt", 900.0) &&
				rangeFt < number("RangeFt", 2200.0) &&
				std::abs(bb->ClosureRate_MS / FT) < number("MaxAbsClosureFtps", 900.0) &&
				bb->ClosureRate_MS / FT > number("OpeningFtps", -99999.0) &&
				bb->Los_Degree > number("MinLosDeg", 58.0) &&
				bb->Los_Degree < number("MaxLosDeg", 105.0) &&
				bb->Los_Degree_Target > number("MinEnemyLOSDeg", 12.0) &&
				bb->Los_Degree_Target < number("MaxEnemyLOSDeg", 55.0) &&
				!immediateOwnCone && !immediateEnemyCone;
			result = bb->BFM == HABFM && (lowEnergyStalemate || pendingNoCue || closeNoWinningCue);
			break;
		}
		case COND_CONTROL_ZONE:
		{
			Optional<bool> approach = getInput<bool>("Approach");
			if (approach && approach.value())
			{
				// Paper block 12 separates "approach the control-zone volume"
				// from "established in control zone". The state is computed once
				// in UpdateGeometry so terminal OBFM adapters cannot redefine it.
				result = bb->BFM == OBFM && bb->ControlZoneState >= 1;
				if (result)
					bb->OffensiveCommitUntil = static_cast<float>(std::max(
						static_cast<double>(bb->OffensiveCommitUntil),
						bb->RunningTime + number("OffensiveCommitSec", 2.0)));
				break;
			}
			result = bb->BFM == OBFM && bb->ControlZoneState == 2;
			if (result)
				bb->OffensiveCommitUntil = static_cast<float>(std::max(
					static_cast<double>(bb->OffensiveCommitUntil),
					bb->RunningTime + number("OffensiveCommitSec", 2.0)));
			break;
		}
		case COND_MAINTAIN_MANEUVER:
		{
			if (bb->LockedManeuverTask < 0)
				break;
			const bool oneCircle = bb->LockedManeuverTask == static_cast<int>(TASK_ONE_CIRCLE);
			const double cueAngle = oneCircle ? 90.0 : 180.0;
			const double maxSeconds = oneCircle
				? number("OneMaxSec", 14.0)
				: number("TwoMaxSec", 24.0);
			const double energyMargin = std::max(
				number("CueEnergyMarginM", 900.0), 600.0);
			const bool ownEnergyOK = bb->EnergyAdvantage_M > -energyMargin;
			const bool enemyEnergyOK = bb->EnergyAdvantage_M < energyMargin;
			const double angularAdvantage =
				static_cast<double>(bb->Los_Degree_Target - bb->Los_Degree);
			const double elapsed = bb->RunningTime - bb->ManeuverStartTime;
			const bool cueMature = bb->ManeuverTurnDegrees > 0.25 * cueAngle || elapsed > 2.5;
			if (bb->LockedManeuverTask == static_cast<int>(TASK_SCISSORS))
			{
				const bool scissorsMode = bb->BFM == HABFM || bb->BFM == DBFM || bb->BFM == SCISSORS;
				if (!scissorsMode)
				{
					resetManeuverLock(bb);
					break;
				}
				const bool keepScissors =
					elapsed < number("ScissorsMaxSec", 5.0) &&
					rangeFt < number("MaxRangeFt", 15000.0) &&
					damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0;
				const bool reformingGunThreat =
					bb->BFM == DBFM &&
					rangeFt < 2600.0 &&
					bb->Los_Degree > 95.0f &&
					bb->Los_Degree_Target < std::max(phaseConeDeg(bb->Phase) + 26.0, 28.0) &&
					bb->EnemyLosRate_DegSec < -8.0f;
				if (keepScissors && !reformingGunThreat)
				{
					result = true;
					break;
				}
				bb->LastManeuverCueOutcome = CUE_NEUTRAL;
				bb->BFM = HABFM;
				const double bridgeEnemyLosDeg = std::max(
					phaseConeDeg(bb->Phase) + number("BridgeGuardDeg", 24.0),
					number("BridgeMinEnemyLOSDeg", 25.0));
				const bool bridgeOutOfScissors =
					rangeFt < number("BridgeRangeFt", 4500.0) &&
					rangeFt > number("BridgeMinRangeFt", 700.0) &&
					bb->Los_Degree > number("BridgeMinMyLOSDeg", 25.0) &&
					bb->Los_Degree < number("BridgeMaxMyLOSDeg", 150.0) &&
					bb->Los_Degree_Target > bridgeEnemyLosDeg &&
					damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0;
				if (bridgeOutOfScissors)
				{
					const double bridgeSec = number("BridgeSec", 1.8);
					bb->HABFMPullToHUDUntil = static_cast<float>(
						std::max(static_cast<double>(bb->HABFMPullToHUDUntil),
							bb->RunningTime + bridgeSec));
					bb->HABFMNextManeuverTask = static_cast<int>(TASK_ONE_CIRCLE);
					bb->ManeuverCooldownUntil = bb->HABFMPullToHUDUntil;
				}
				else
					bb->ManeuverCooldownUntil = static_cast<float>(bb->RunningTime + 0.8);
				resetManeuverLock(bb);
				break;
			}
			if (bb->LockedManeuverTask == static_cast<int>(TASK_BREAK_JINK))
			{
				const bool enemyActualCone =
					damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) > 0.0;
				const bool closeRearGunDefense =
					bb->BFM == DBFM &&
					elapsed < std::max(number("BreakJinkMaxSec", 0.75), 1.20) &&
					rangeFt < 2000.0 &&
					bb->Los_Degree > 125.0f &&
					bb->Los_Degree_Target < 18.0f &&
					damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0;
				const bool keepBreak =
					bb->BFM == DBFM &&
					rangeFt < number("MaxRangeFt", 7000.0) &&
					(closeRearGunDefense ||
					 (elapsed < number("BreakJinkMaxSec", 0.75) &&
					  (enemyActualCone ||
					   bb->ThreatClearTime < number("MinThreatClearSec", 0.45) ||
					   bb->Los_Degree_Target < number("BreakJinkEnemyLOSDeg", 42.0))));
				if (keepBreak)
				{
					result = true;
					break;
				}
				resetManeuverLock(bb);
				break;
			}
			if (bb->BFM != HABFM)
			{
				resetManeuverLock(bb);
				break;
			}

			// The paper re-evaluates OBFM/HABFM/DBFM every tick. A maneuver lock
			// may hold the selected circle, but it must yield when a clear 3/9-line
			// advantage or disadvantage has already formed.
			const bool strongLosingGeometry = cueMature && rangeFt < 8500.0 &&
				bb->Los_Degree_Target < 45.0f && bb->Los_Degree > 90.0f &&
				angularAdvantage < -25.0 && enemyEnergyOK;
			if (strongLosingGeometry)
			{
				bb->LastManeuverCueOutcome = CUE_STRONG_LOSE;
				bb->BFM = DBFM;
				bb->DefensiveCommitUntil = static_cast<float>(std::max(
					static_cast<double>(bb->DefensiveCommitUntil), bb->RunningTime + 1.2));
				bb->ManeuverCooldownUntil = static_cast<float>(bb->RunningTime + 1.0);
				resetManeuverLock(bb);
				break;
			}
			const bool strongWinningGeometry = cueMature && rangeFt < 8500.0 &&
				bb->Los_Degree < 70.0f && bb->Los_Degree_Target > 95.0f &&
				angularAdvantage > 25.0 && ownEnergyOK;
			if (strongWinningGeometry)
			{
				bb->LastManeuverCueOutcome = CUE_STRONG_WIN;
				bb->BFM = OBFM;
				armNode35(bb, 5.5);
				bb->OffensiveCommitUntil = static_cast<float>(bb->RunningTime +
					number("OffensiveCommitSec", 5.0));
				bb->ManeuverCooldownUntil = static_cast<float>(bb->RunningTime + 1.0);
				resetManeuverLock(bb);
				break;
			}
			const double noCueSec = number("NoCueSec", oneCircle ? 7.0 : 8.0);
			const bool neutralNoCue =
				oneCircle &&
				cueMature &&
				elapsed > noCueSec &&
				isCloseLowEnergyStalemate(bb, rangeFt, number("NeutralSec", 15.0)) &&
				std::abs(angularAdvantage) < 15.0 &&
				std::abs(static_cast<double>(bb->EnergyAdvantage_M)) < energyMargin &&
				bb->Los_Degree > 25.0f &&
				bb->Los_Degree_Target > 25.0f;
			if (neutralNoCue)
			{
				bb->LastManeuverCueOutcome = CUE_NEUTRAL;
				// Paper block 21: Scissors is a close, low-energy neutral
				// stalemate response. Two-circle neutral at the 180-deg cue is
				// handled below by Pull-to-HUD -> OneCircle, not by far scissors.
				bb->BFM = HABFM;
				bb->ManeuverCooldownUntil = static_cast<float>(bb->RunningTime + 0.2);
				bb->PendingScissors = true;
				resetManeuverLock(bb);
				break;
			}

			const bool timeoutCue = elapsed > maxSeconds;
			const bool rangeAbort = rangeFt > number("MaxRangeFt", 15000.0);
			if (bb->ManeuverTurnDegrees < cueAngle && !timeoutCue && !rangeAbort)
			{
				result = true;
				break;
			}
			if (rangeAbort && bb->ManeuverTurnDegrees < cueAngle)
			{
				bb->LastManeuverCueOutcome = CUE_RANGE_ABORT;
				bb->BFM = HABFM;
				bb->ManeuverCooldownUntil = static_cast<float>(bb->RunningTime + 0.8);
				resetManeuverLock(bb);
				break;
			}
			if (timeoutCue && bb->ManeuverTurnDegrees < cueAngle)
			{
				bb->LastManeuverCueOutcome = CUE_TIMEOUT;
				bb->BFM = HABFM;
				bb->ManeuverCooldownUntil = static_cast<float>(bb->RunningTime + 0.8);
				resetManeuverLock(bb);
				break;
			}

			if (oneCircle)
			{
				// Paper Fig. 4c: target in front of own 3/9 line plus sufficient
				// relative energy is the one-circle winning cue; then reverse into
				// the two-circle rate fight. Relative angular advantage replaces the
				// unavailable adversary EM-chart query.
				const bool adversaryInFrontOf39 = bb->Los_Degree < 90.0f;
				const bool winningCue = adversaryInFrontOf39 &&
					angularAdvantage > 10.0 && ownEnergyOK;
				const bool losingCue = !adversaryInFrontOf39 &&
					angularAdvantage < -15.0 && enemyEnergyOK;
				if (winningCue)
				{
					const double nextSide = -bb->LockedManeuverSide;
					bb->LastManeuverCueOutcome = CUE_WIN;
					startManeuverLock(bb, static_cast<int>(TASK_TWO_CIRCLE), nextSide);
					result = true;
					break;
				}
				if (losingCue)
				{
					bb->LastManeuverCueOutcome = CUE_LOSE;
					bb->BFM = DBFM;
					bb->DefensiveCommitUntil = static_cast<float>(std::max(
						static_cast<double>(bb->DefensiveCommitUntil), bb->RunningTime + 1.0));
					bb->ManeuverCooldownUntil = static_cast<float>(bb->RunningTime + 1.0);
				}
				else
				{
					bb->LastManeuverCueOutcome = CUE_NEUTRAL;
					bb->BFM = HABFM;
					bb->PendingScissors = true;
					bb->ManeuverCooldownUntil = static_cast<float>(bb->RunningTime + 0.8);
				}
			}
			else
			{
				// Paper Fig. 4d: two-circle 180-deg cue.  A clear angular
				// advantage plus non-losing energy enters OBFM; a clear disadvantage
				// enters DBFM. The remaining neutral case uses block 18 Pull-to-HUD
				// and then starts one-circle.
				const bool winningCue = angularAdvantage > 15.0 && ownEnergyOK &&
					bb->Los_Degree < 90.0f && bb->Los_Degree_Target > 90.0f;
				const bool losingCue = angularAdvantage < -15.0 && enemyEnergyOK;
				if (winningCue)
				{
					bb->LastManeuverCueOutcome = CUE_WIN;
					bb->BFM = OBFM;
					armNode35(bb, 5.5);
					bb->OffensiveCommitUntil = static_cast<float>(bb->RunningTime +
						number("OffensiveCommitSec", 5.0));
					bb->ManeuverCooldownUntil = static_cast<float>(bb->RunningTime + 1.0);
				}
				else if (losingCue)
				{
					bb->LastManeuverCueOutcome = CUE_LOSE;
					bb->BFM = DBFM;
					bb->DefensiveCommitUntil = static_cast<float>(std::max(
						static_cast<double>(bb->DefensiveCommitUntil), bb->RunningTime + 1.0));
					bb->ManeuverCooldownUntil = static_cast<float>(bb->RunningTime + 1.0);
				}
				else
				{
					bb->LastManeuverCueOutcome = CUE_NEUTRAL;
					bb->BFM = HABFM;
					// Supplemental flow: a non-winning two-circle always goes
					// through block 18 (Pull to HUD) and then block 19
					// (OneCircle).  The published edge has no 6000-ft cutoff;
					// that cutoff diverted neutral rate fights into the non-paper
					// ArcRejoin fallback.  Allow enough time for a far 180-deg
					// completion to put the adversary on the HUD; higher-priority
					// HABFM cone-employment cues can still pre-empt this bridge.
					bb->HABFMPullToHUDUntil = static_cast<float>(bb->RunningTime + 10.0);
					bb->HABFMNextManeuverTask = static_cast<int>(TASK_ONE_CIRCLE);
					bb->ManeuverCooldownUntil = bb->HABFMPullToHUDUntil;
					bb->PendingScissors = false;
				}
			}
			resetManeuverLock(bb);
			break;
		}
		case COND_THREAT_CLEARED:
		{
			// Leave DBFM only after the adversary has lost both the near-cone and
			// the clear angular advantage. This prevents the old DBFM -> Pure
			// fallback while the bandit is still established behind ownship.
			const double angularAdvantage =
				static_cast<double>(bb->Los_Degree_Target - bb->Los_Degree);
			const bool standardClear = bb->ThreatClearTime > number("MinSec", 0.8) &&
				bb->Los_Degree_Target > number("MinEnemyLOSDeg", 80.0) &&
				angularAdvantage > -20.0;
			// Paper DBFM is not a permanent defensive orbit: after the break has
			// neutralized the pure-pursuit attacker into a low-closure tail chase,
			// pull back to the HUD/reversal flow instead of holding HardTurn.
			const bool defensiveLureComplete =
				bb->BFM == DBFM &&
				bb->ThreatClearTime > number("MinSec", 0.8) &&
				rangeFt > number("MinRangeFt", 1300.0) &&
				rangeFt < number("RangeFt", 6500.0) &&
				std::abs(bb->ClosureRate_MS / FT) < number("ClosureFtps", 520.0) &&
				bb->Los_Degree > number("MinLosDeg", 92.0) &&
				bb->Los_Degree_Target > number("MinClearEnemyLOSDeg", 70.0) &&
				bb->Los_Degree_Target < number("EnemyLOSDeg", 125.0) &&
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0;
			const double closingFtps = std::max(0.0, -static_cast<double>(bb->ClosureRate_MS) / FT);
			const bool positionalThreatCleared =
				bb->Los_Degree_Target > number("MinClearEnemyLOSDeg", 55.0) &&
				angularAdvantage > -30.0;
			const bool contestWezClear =
				bb->BFM == DBFM &&
				bb->ThreatClearTime > std::max(number("MinSec", 0.8), 0.75) &&
				rangeFt > number("MinRangeFt", 1300.0) &&
				rangeFt < number("RangeFt", 6500.0) &&
				bb->Los_Degree_Target > phaseConeDeg(bb->Phase) + number("ContestGuardDeg", 5.0) &&
				positionalThreatCleared &&
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0 &&
				(closingFtps < number("ClosureFtps", 520.0) * 0.40 ||
				 bb->ClosureRate_MS / FT > 0.0 ||
				 rangeFt > phaseMaxRangeFt(bb->Phase) + number("RangeBufFt", 800.0));
			const bool lowClosurePureResolved =
				bb->BFM == DBFM &&
				bb->ThreatClearTime > std::max(number("MinSec", 0.8), 1.2) &&
				bb->EnemyPursuitType == EPT_PURE &&
				rangeFt > number("MinRangeFt", 1300.0) &&
				rangeFt < number("RangeFt", 6500.0) &&
				bb->Los_Degree_Target > phaseConeDeg(bb->Phase) + 8.0 &&
				positionalThreatCleared &&
				std::abs(bb->ClosureRate_MS / FT) < number("PureClearMaxAbsClosureFtps", 140.0) &&
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0;
			const bool counterCue =
				bb->Los_Degree < number("CounterMaxMyLOSDeg", 105.0) &&
				bb->Los_Degree_Target > number("CounterMinEnemyLOSDeg", 65.0) &&
				angularAdvantage > number("CounterMinAngularAdvDeg", -5.0) &&
				rangeFt < number("CounterMaxRangeFt", 6500.0) &&
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0;
			if (standardClear || defensiveLureComplete || contestWezClear || lowClosurePureResolved)
			{
				const double recoverySec = number("RecoverySec", 0.45);
				if (counterCue)
				{
					bb->BFM = OBFM;
					bb->OffensiveCommitUntil = static_cast<float>(std::max(
						static_cast<double>(bb->OffensiveCommitUntil), bb->RunningTime + 3.0));
					const double node35Sec = number("Node35Sec", 0.0);
					if (node35Sec > 0.0)
						armNode35(bb, node35Sec);
				}
				else
				{
					bb->BFM = HABFM;
				}
				bb->DefensiveRecoveryUntil = static_cast<float>(std::max(
					static_cast<double>(bb->DefensiveRecoveryUntil),
					bb->RunningTime + recoverySec));
				bb->DefensiveCommitUntil = static_cast<float>(bb->RunningTime);
			}
			result = standardClear || defensiveLureComplete || contestWezClear || lowClosurePureResolved;
			break;
		}
		case COND_ROOM_TO_MANEUVER:
			// Paper blocks 14/16 are an OR path: room > 3000 ft goes directly
			// to two-circle; when room is insufficient, high EM energy also
			// selects two-circle, otherwise one-circle.  The previous AND gate
			// did not implement the published flow.
			result = rangeFt > number("RangeFt", 3000.0) || hasHighManeuverEnergy(bb);
			break;
		case COND_OUTSIDE_TURN_CIRCLE:
		{
			// BEM block 11: approach the tangential entry window before
			// following the hostile inside its turn circle.
			if (!bb->TargetTurnCircleValid)
				break;
			const BT_Geometry::Vector3 targetRadial =
				bb->TargetLocaion_Cartesian - bb->TargetTurnCenter;
			BT_Geometry::Vector3 turnAxis = targetRadial.cross(bb->TargetVelocity);
			if (turnAxis.length() <= 1.0)
				break;
			turnAxis.normalize();
			const BT_Geometry::Vector3 centerToOwn =
				bb->MyLocation_Cartesian - bb->TargetTurnCenter;
			const BT_Geometry::Vector3 ownInTurnPlane =
				centerToOwn - turnAxis * centerToOwn.dot(turnAxis);
			// VER08: the finite-difference circle estimate can remain "outside"
			// while a close control-zone or cone capture is already available.
			// Do not let LagEntry repeatedly steal those close OBFM ticks.
			const bool closeCaptureGeometry =
				rangeFt < 4300.0 &&
				bb->Los_Degree < 70.0f &&
				bb->Los_Degree_Target > 45.0f;
			result = rangeFt < number("MaxRangeFt", 6500.0) &&
				!closeCaptureGeometry &&
				ownInTurnPlane.length() >
				bb->TargetTurnRadius_M + number("RangeBufFt", 500.0) * FT;
			break;
		}
		case COND_LOW_ENERGY:
		{
			// Energy recovery must be a safe far reset, not a default neutral
			// behavior. VER03 spent 30% of runtime here and lost shot windows.
			const double rangeBuf = number("RangeBufFt", 1200.0);
			const double myPotential = nearConePotential(bb->Los_Degree, rangeFt, bb->Phase, 7.0, rangeBuf);
			const double enemyPotential = nearConePotential(bb->Los_Degree_Target, rangeFt, bb->Phase, 6.0, rangeBuf);
			result = bb->BFM == HABFM &&
				bb->MyKCAS_KT > 0.0f &&
				bb->MyKCAS_KT < number("EnterKCAS", 235.0) &&
				rangeFt > maxRangeFt + rangeBuf &&
				bb->Los_Degree > number("MinMyLOSDeg", 14.0) &&
				bb->Los_Degree_Target > number("MinEnemyLOSDeg", 14.0) &&
				myPotential < 0.08 && enemyPotential < 0.10 &&
				bb->RunningTime > bb->ShotCommitUntil &&
				bb->RunningTime > bb->OffensiveCommitUntil &&
				bb->RunningTime > bb->DefensiveCommitUntil &&
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0;
			break;
		}

		case COND_CROSSING_CONE_LEAD:
		{
			// Competition LeadVPP gate. Unlike the paper's bullet-TOF LeadVPP,
			// this is used when the target is near the damage cone but crossing
			// fast enough that pure pursuit will trail the cone center. Stable
			// targets already inside the strict phase cone fall through to
			// Task_Track, and enemy near-cone superiority still yields DBFM.
			const double rate = apparentLosRateDegSec(bb);
			const double leadGateDeg = number("MaxLosDeg", std::max(10.0, 3.0 * coneDeg + 6.0));
			const double minLeadLos = number("MinLosDeg", 0.0);
			const double minEnemyLos = number("MinEnemyLOSDeg", 0.0);
			const double minRate = number("MinAbsLosRateDegps", 1.5);
			const double maxRange = number("MaxRangeFt", maxRangeFt + 800.0);
			const double minAngularAdvDeg = number("MinAngularAdvDeg", -9999.0);
			const double angularAdvantage =
				static_cast<double>(bb->Los_Degree_Target - bb->Los_Degree);
			const bool committedPureTrack =
				bb->RunningTime < bb->ShotCommitUntil &&
				bb->Los_Degree < number("ShotCommitMaxLosDeg", 12.0) &&
				bb->Los_Degree_Target > number("MinEnemyLOSDeg", 25.0);
			const bool stableInCone = damageRate(bb->Los_Degree, rangeFt, bb->Phase) > 0.0 &&
				rate < std::max(0.60, 0.45 * minRate);
			const bool enemyHasBetterNearWez = enemyHasBetterNearCone(
				bb, rangeFt, coneDeg, maxRangeFt, 4.0, number("EnemyBetterMarginDeg", 1.0));
			result = rangeFt > number("MinRangeFt", 600.0) &&
				rangeFt < maxRange &&
				bb->Los_Degree > minLeadLos &&
				bb->Los_Degree < leadGateDeg &&
				bb->MyKCAS_KT > number("MinKCAS", 0.0) &&
				bb->MyKCAS_KT < number("MaxKCAS", 9999.0) &&
				bb->Los_Degree_Target > minEnemyLos &&
				angularAdvantage > minAngularAdvDeg &&
				rate > minRate &&
				!committedPureTrack &&
				!stableInCone &&
				!enemyHasBetterNearWez;
			if (result)
				bb->ShotCommitUntil = static_cast<float>(std::max(static_cast<double>(bb->ShotCommitUntil),
					bb->RunningTime + number("ShotCommitSec", 0.9)));
			break;
		}
		case COND_NEAR_CONE_AIM:
		{
			// Shot-conversion gate: begin mostly-pure aiming before the strict
			// phase cone is already satisfied. This is what was missing in VER02:
			// Track appeared almost never, so baseline HP stayed at 1.0.
			const double maxLos = number("MaxLosDeg", std::max(7.0, coneDeg + 6.0));
			Optional<double> phaseRangeBuffer = getInput<double>("MaxRangeBufferFt");
			const double maxRange = phaseRangeBuffer
				? maxRangeFt + phaseRangeBuffer.value()
				: number("MaxRangeFt", maxRangeFt + 700.0);
			const double maxAbsLosRate = number("MaxAbsLosRateDegps", 9999.0);
			const double absLosRate = apparentLosRateDegSec(bb);
			const double rangeBuf = 800.0;
			const double myPotential = nearConePotential(bb->Los_Degree, rangeFt, bb->Phase, maxLos - coneDeg, rangeBuf);
			const double enemyPotential = nearConePotential(bb->Los_Degree_Target, rangeFt, bb->Phase,
				number("MaxEnemyLOSDeg", std::max(6.0, coneDeg + 4.0)) - coneDeg, rangeBuf);
			const bool enemyClearlyBetter =
				bb->Los_Degree_Target + number("EnemyBetterMarginDeg", 1.0) < bb->Los_Degree &&
				bb->Los_Degree_Target < number("MaxEnemyLOSDeg", std::max(6.0, coneDeg + 4.0)) &&
				inRangeBand(rangeFt, maxRangeFt, rangeBuf) &&
				(enemyPotential > myPotential + number("DamageMargin", 0.03) ||
				 enemyHasBetterNearCone(bb, rangeFt, coneDeg, maxRangeFt, 5.0,
					number("EnemyBetterMarginDeg", 1.0)));
			const double closingFtps = std::max(0.0, -static_cast<double>(bb->ClosureRate_MS) / FT);
			const bool terminalOffensivePass =
				bb->BFM != DBFM &&
				rangeFt < maxRangeFt + 1000.0 &&
				bb->Los_Degree < std::min(maxLos, 30.0) &&
				bb->Los_Degree_Target > coneDeg + 5.0 &&
				closingFtps > 650.0 &&
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0;
			const bool preRangeHighClosureLeadNeeded =
				bb->Phase <= 1 &&
				rangeFt > maxRangeFt + 550.0 &&
				closingFtps > 750.0 &&
				bb->Los_Degree > std::max(14.0, coneDeg + 6.0) &&
				bb->Los_Degree_Target < 20.0f &&
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0;
			const bool lowAspectHighClosureLeadNeeded =
				bb->Phase <= 1 &&
				rangeFt > maxRangeFt - 850.0 &&
				rangeFt < maxRangeFt + 900.0 &&
				closingFtps > 750.0 &&
				bb->Los_Degree > std::max(6.5, coneDeg + 5.0) &&
				bb->Los_Degree_Target < 15.0f &&
				damageRate(bb->Los_Degree, rangeFt, bb->Phase) <= 0.0 &&
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0;
			result = bb->Phase >= number("MinPhase", 1.0) &&
				bb->Los_Degree < maxLos &&
				bb->Los_Degree > number("MinLosDeg", 0.0) &&
				rangeFt > number("MinRangeFt", 550.0) &&
				rangeFt < maxRange &&
				bb->MyKCAS_KT > number("MinKCAS", 0.0) &&
				bb->MyKCAS_KT < number("MaxKCAS", 9999.0) &&
				absLosRate < maxAbsLosRate &&
				bb->Los_Degree_Target > number("MinEnemyLOSDeg", 0.0) &&
				bb->Los_Degree_Target < number("MaxAllowedEnemyLOSDeg", 9999.0) &&
				!preRangeHighClosureLeadNeeded &&
				!lowAspectHighClosureLeadNeeded &&
				!weakOpeningEdgeShot(bb, rangeFt, maxRangeFt, coneDeg) &&
				(!enemyClearlyBetter || terminalOffensivePass);
			if (result)
			{
				bb->ShotCommitUntil = static_cast<float>(std::max(static_cast<double>(bb->ShotCommitUntil),
					bb->RunningTime + number("ShotCommitSec", 1.2)));
				const double offensiveCommitSec = number("OffensiveCommitSec", 0.0);
				if (offensiveCommitSec > 0.0)
					bb->OffensiveCommitUntil = static_cast<float>(std::max(static_cast<double>(bb->OffensiveCommitUntil),
						bb->RunningTime + offensiveCommitSec));
			}
			break;
		}
		case COND_PENDING_SCISSORS:
		{
			// Restore the paper's HABFM no-winning-cue -> block 21 Scissors
			// connection, but only for the close/low-energy stalemate described
			// in the supplemental flow. Far rate-fight no-cue remains HABFM.
			const bool myActualCone = damageRate(bb->Los_Degree, rangeFt, bb->Phase) > 0.0;
			const bool enemyActualCone = damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) > 0.0;
			result = bb->PendingScissors && bb->BFM == HABFM &&
				isCloseLowEnergyStalemate(bb, rangeFt, number("MinNeutralSec", 15.0)) &&
				rangeFt < number("MaxRangeFt", 3000.0) &&
				!myActualCone && !enemyActualCone;
			if (result)
			{
				startManeuverLock(bb, static_cast<int>(TASK_SCISSORS), turnSide(bb));
				bb->PendingScissors = false;
			}
			break;
		}
		case COND_CONTROL_ZONE_CAPTURE:
		{
			const bool enemyActualCone =
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) > 0.0;
			const double angularAdvantage =
				static_cast<double>(bb->Los_Degree_Target - bb->Los_Degree);
			const bool readyForTerminalAim =
				bb->Los_Degree < 28.0f &&
				rangeFt < phaseMaxRangeFt(bb->Phase) + 900.0 &&
				std::abs(bb->ClosureRate_MS / FT) < 650.0;
			const bool closeWideOvershoot =
				rangeFt < 1700.0 &&
				bb->Los_Degree > 45.0f;
			const bool strictCapture = bb->BFM == OBFM &&
				rangeFt > number("MinRangeFt", 900.0) &&
				rangeFt < number("MaxRangeFt", 4600.0) &&
				bb->Los_Degree > number("MinLosDeg", 0.0) &&
				bb->Los_Degree < number("MaxLosDeg", 75.0) &&
				bb->Los_Degree_Target > number("MinEnemyLOSDeg", 65.0) &&
				std::abs(bb->ClosureRate_MS / FT) < number("MaxClosureFtps", 900.0) &&
				angularAdvantage > number("MinAngularAdvDeg", 10.0) &&
				!enemyActualCone &&
				!readyForTerminalAim &&
				!closeWideOvershoot;
			const bool wasCaptureTask =
				bb->ActiveCompetitionTask == static_cast<int>(TASK_PULL_TO_HUD) ||
				bb->ActiveCompetitionTask == static_cast<int>(TASK_CONTROL_ZONE) ||
				bb->ActiveCompetitionTask == static_cast<int>(TASK_FOLLOW_HOSTILE) ||
				bb->ActiveCompetitionTask == static_cast<int>(TASK_TRACK) ||
				bb->ActiveCompetitionTask == static_cast<int>(TASK_CONE_LEAD_TRACK);
			const bool committedCapture = bb->BFM == OBFM &&
				wasCaptureTask &&
				bb->RunningTime < bb->OffensiveCommitUntil &&
				rangeFt > 700.0 &&
				rangeFt < 6200.0 &&
				bb->Los_Degree < 95.0f &&
				bb->Los_Degree_Target > 35.0f &&
				std::abs(bb->ClosureRate_MS / FT) < 1900.0 &&
				!enemyActualCone &&
				!readyForTerminalAim &&
				!closeWideOvershoot;
			result = strictCapture || committedCapture;
			if (result)
				bb->OffensiveCommitUntil = static_cast<float>(std::max(
					static_cast<double>(bb->OffensiveCommitUntil),
					bb->RunningTime + number("OffensiveCommitSec", 2.5)));
			break;
		}
		case COND_HABFM_BRIDGE:
		{
			if (bb->HABFMNextManeuverTask < 0)
				break;
			if (bb->BFM != HABFM)
			{
				bb->HABFMNextManeuverTask = -1;
				bb->HABFMPullToHUDUntil = 0.0f;
				break;
			}
			const double bridgeClosingFtps = std::max(0.0,
				-static_cast<double>(bb->ClosureRate_MS) / FT);
			const bool enemyActualCone =
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) > 0.0;
			// Block 18 says "Pull to HUD", which is a geometric completion
			// condition rather than a fixed-duration command.  If the initial
			// timeout expires while ownship still owns a converging HUD capture,
			// keep the bridge alive through the phase scoring range instead of
			// reversing into OneCircle just before employment.
			const bool ownsFineHudCapture =
				bb->Los_Degree < 10.0f &&
				bb->Los_Degree <= bb->Los_Degree_Target + 1.0f;
			const bool ownsBroadHudCapture =
				bb->Los_Degree < 30.0f &&
				bb->Los_Degree_Target > bb->Los_Degree + 5.0f;
			const bool hudCaptureConverging =
				rangeFt > 550.0 &&
				rangeFt < phaseMaxRangeFt(bb->Phase) + 6000.0 &&
				bridgeClosingFtps > 300.0 &&
				bb->Los_Degree_Target > phaseConeDeg(bb->Phase) + 0.25f &&
				(ownsFineHudCapture || ownsBroadHudCapture) &&
				!enemyActualCone;
			// A maneuver block also needs a geometric failure edge.  If the
			// adversary owns the smaller nose angle and our LOS is opening inside
			// the weapon-conversion corridor, Pull-to-HUD has failed; the
			// supplemental flow proceeds to its queued OneCircle rather than
			// blindly honoring the initial timeout.
			const bool hudCaptureLost =
				rangeFt > 550.0 &&
				rangeFt < phaseMaxRangeFt(bb->Phase) + 2000.0 &&
				bb->Los_Degree > 15.0f &&
				bb->Los_Degree > bb->Los_Degree_Target + 6.0f &&
				bb->MyLosRate_DegSec > 1.0f;
			// The HABFM subtree's crossing-lead sequence is the local Employ
			// handoff for a target sweeping across the HUD.  Let that sequence own
			// the tick only when ownship does not already own the angular geometry;
			// otherwise the authoritative block-18 pull remains uninterrupted.
			const bool allowCrossingEmployHandoff =
				bb->RunningTime < bb->HABFMPullToHUDUntil &&
				rangeFt > 700.0 && rangeFt < 4800.0 &&
				bb->Los_Degree >= 8.0f && bb->Los_Degree <= 28.0f &&
				std::abs(apparentLosRateDegSec(bb)) >= 5.0 &&
				bb->Los_Degree_Target >= 7.0f &&
				bb->Los_Degree <= bb->Los_Degree_Target + 8.0f &&
				bb->Los_Degree >= bb->Los_Degree_Target;
			if (allowCrossingEmployHandoff)
			{
				result = false;
				break;
			}
			if (!hudCaptureLost &&
				(bb->RunningTime < bb->HABFMPullToHUDUntil || hudCaptureConverging))
			{
				if (hudCaptureConverging)
					bb->HABFMPullToHUDUntil = static_cast<float>(std::max(
						static_cast<double>(bb->HABFMPullToHUDUntil), bb->RunningTime + 0.5));
				result = true;
				break;
			}
			if (bb->LockedManeuverTask < 0)
			{
				const int nextTask = bb->HABFMNextManeuverTask;
				const double hostileSide = targetTurnSide(bb);
				// Paper: one-circle uses opposite WORLD turn directions; two-circle
				// uses the same world turn direction.
				const double nextSide = nextTask == static_cast<int>(TASK_ONE_CIRCLE)
					? -hostileSide : hostileSide;
				startManeuverLock(bb, nextTask, nextSide);
			}
			bb->HABFMNextManeuverTask = -1;
			bb->HABFMPullToHUDUntil = 0.0f;
			bb->ManeuverCooldownUntil = static_cast<float>(bb->RunningTime + 0.2);
			// Return success for this tick so the PostMerge selector cannot replace
			// the newly-created lock. MaintainManeuver owns the next tick.
			result = true;
			break;
		}
		case COND_ENEMY_PURE_THREAT:
		{
			// DBFM supplement from the paper: identify an attacker that is not
			// merely in the WEZ yet, but is flying a pure-pursuit collision line.
			// This lets DBFM lure the baseline into overshoot instead of only
			// reacting after strict cone damage begins.
			const bool ownActualGun = damageRate(bb->Los_Degree, rangeFt, bb->Phase) > 0.0;
			const double closingFtps = std::max(0.0, -static_cast<double>(bb->ClosureRate_MS) / FT);
			result = bb->BFM == DBFM &&
				bb->EnemyPursuitType == EPT_PURE &&
				rangeFt > number("MinRangeFt", 900.0) &&
				rangeFt < number("MaxRangeFt", 6500.0) &&
				closingFtps >= number("MinClosureFtps", 0.0) &&
				closingFtps <= number("MaxClosureFtps", 99999.0) &&
				bb->Los_Degree > number("MinMyLOSDeg", 35.0) &&
				!ownActualGun;
			if (result)
			{
				bb->LockedManeuverTask = -1;
				bb->DefensiveCommitUntil = static_cast<float>(std::max(
					static_cast<double>(bb->DefensiveCommitUntil),
					bb->RunningTime + number("CommitSec", 1.2)));
				Optional<bool> breakJinkLock = getInput<bool>("BreakJinkLock");
				if (breakJinkLock && breakJinkLock.value())
					startManeuverLock(bb, static_cast<int>(TASK_BREAK_JINK), turnSide(bb));
			}
			break;
		}
		case COND_ENEMY_LEAD_THREAT:
		{
			const bool ownActualGun = damageRate(bb->Los_Degree, rangeFt, bb->Phase) > 0.0;
			result = bb->BFM == DBFM &&
				bb->EnemyPursuitType == EPT_LEAD &&
				rangeFt > number("MinRangeFt", 600.0) &&
				rangeFt < number("MaxRangeFt", maxRangeFt + 1200.0) &&
				bb->Los_Degree > number("MinMyLOSDeg", 45.0) &&
				!ownActualGun;
			if (result)
			{
				bb->DefensiveCommitUntil = static_cast<float>(std::max(
					static_cast<double>(bb->DefensiveCommitUntil),
					bb->RunningTime + number("CommitSec", 1.0)));
				Optional<bool> breakJinkLock = getInput<bool>("BreakJinkLock");
				if (breakJinkLock && breakJinkLock.value())
					startManeuverLock(bb, static_cast<int>(TASK_BREAK_JINK), turnSide(bb));
			}
			break;
		}
		case COND_ENEMY_LAG_PURSUIT:
		{
			const bool enemyActualCone = damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) > 0.0;
			result = bb->BFM == DBFM &&
				bb->EnemyPursuitType == EPT_LAG &&
				bb->ThreatClearTime > number("MinClearSec", 0.5) &&
				rangeFt > number("MinRangeFt", 1200.0) &&
				rangeFt < number("MaxRangeFt", 7000.0) &&
				!enemyActualCone;
			if (result)
				bb->DefensiveCommitUntil = static_cast<float>(std::max(
					static_cast<double>(bb->DefensiveCommitUntil),
					bb->RunningTime + number("CommitSec", 0.8)));
			break;
		}
		case COND_DEFENSIVE_COUNTER_CUE:
		{
			const double angularAdvantage =
				static_cast<double>(bb->Los_Degree_Target - bb->Los_Degree);
			const bool enemyActualCone = damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) > 0.0;
			result = bb->BFM == DBFM &&
				bb->ThreatClearTime > number("MinSec", 0.6) &&
				rangeFt > number("MinRangeFt", 1000.0) &&
				rangeFt < number("MaxRangeFt", 6500.0) &&
				bb->Los_Degree < number("MaxMyLOSDeg", 105.0) &&
				bb->Los_Degree_Target > number("MinEnemyLOSDeg", 65.0) &&
				angularAdvantage > number("MinAngularAdvDeg", -5.0) &&
				!enemyActualCone;
			if (result)
			{
				bb->BFM = OBFM;
				bb->DefensiveCommitUntil = static_cast<float>(bb->RunningTime);
				bb->OffensiveCommitUntil = static_cast<float>(std::max(
					static_cast<double>(bb->OffensiveCommitUntil),
					bb->RunningTime + number("OffensiveCommitSec", 2.0)));
				const double node35Sec = number("Node35Sec", 0.0);
				if (node35Sec > 0.0)
					armNode35(bb, node35Sec);
			}
			break;
		}
		case COND_NODE35_ACTIVE:
		{
			const bool enemyActualCone = damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) > 0.0;
			result = bb->BFM == OBFM &&
				bb->Node35State > 0 &&
				bb->ControlZoneState < 2 &&
				bb->RunningTime < bb->Node35Until &&
				rangeFt > number("MinRangeFt", 900.0) &&
				rangeFt < number("MaxRangeFt", 6500.0) &&
				!enemyActualCone;
			if (result)
				bb->OffensiveCommitUntil = static_cast<float>(std::max(
					static_cast<double>(bb->OffensiveCommitUntil),
					bb->RunningTime + number("OffensiveCommitSec", 2.5)));
			break;
		}
		case COND_SHOT_COMMIT:
		{
			// Hold Track briefly after a genuine near-cone acquisition. Wide
			// pull-to-HUD motion belongs to FollowingHostile/PurePursuit; keeping
			// this gate narrow lets crossing LeadVPP and closure management run.
			const double maxLos = number("MaxLosDeg", std::max(80.0, coneDeg + 60.0));
			const double maxRange = number("MaxRangeFt", maxRangeFt + 2200.0);
			const double closingFtps = std::max(0.0, -static_cast<double>(bb->ClosureRate_MS) / FT);
			const bool enemyActualCone = damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) > 0.0;
			const bool lowAspectHighClosureHandoff =
				bb->Phase <= 1 &&
				rangeFt > maxRangeFt - 850.0 &&
				rangeFt < maxRangeFt + 900.0 &&
				closingFtps > 750.0 &&
				bb->Los_Degree > std::max(6.5, coneDeg + 5.0) &&
				bb->Los_Degree_Target < 15.0f &&
				damageRate(bb->Los_Degree, rangeFt, bb->Phase) <= 0.0;
			result = bb->RunningTime < bb->ShotCommitUntil &&
				rangeFt > 500.0 && rangeFt < maxRange &&
				bb->Los_Degree < maxLos &&
				bb->Los_Degree_Target > number("MinEnemyLOSDeg", 0.0) &&
				bb->Los_Degree_Target < number("MaxEnemyLOSDeg", 180.0) &&
				!enemyActualCone &&
				!lowAspectHighClosureHandoff;
			break;
		}
		}
		const uint32_t conditionBit = uint32_t(1) << static_cast<uint32_t>(kind_);
		bb->EvaluatedConditionMask |= conditionBit;
		if (result)
			bb->TrueConditionMask |= conditionBit;
		return result ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
	}

	CompetitionTask::CompetitionTask(const std::string& name, const NodeConfiguration& config,
		CompetitionTaskKind kind) : CompetitionNode(name, config), kind_(kind) {}

	NodeStatus CompetitionTask::tick()
	{
		CPPBlackBoard* bb = board();
		bb->ActiveCompetitionTask = static_cast<int>(kind_);
		CompetitionTaskKind commandKind = kind_;
		if (kind_ == TASK_MAINTAIN_MANEUVER && bb->LockedManeuverTask >= 0)
			commandKind = static_cast<CompetitionTaskKind>(bb->LockedManeuverTask);
		bb->TrackSubMode = TRACK_NONE;
		bb->VPPMode = VPP_NONE;
		bb->TrackSubReason = TRACK_REASON_NONE;
		bb->ThrottleReason = THROTTLE_REASON_DEFAULT;
		if (bb->PreviousVppTaskKind != static_cast<int>(commandKind))
		{
			double initialWeight = 0.50;
			switch (commandKind)
			{
			case TASK_NODE35_EXTENDED_SIX:
			case TASK_LAG_ENTRY:
				initialWeight = 0.28;
				break;
			case TASK_FOLLOW_HOSTILE:
			case TASK_CONTROL_ZONE:
				initialWeight = 0.45;
				break;
			case TASK_TRACK:
			case TASK_CONE_LEAD_TRACK:
				initialWeight = 0.55;
				break;
			default:
				initialWeight = 0.50;
				break;
			}
			bb->VppBlendWeight = static_cast<float>(initialWeight);
			bb->PreviousVppTaskKind = static_cast<int>(commandKind);
		}
		if ((kind_ == TASK_ONE_CIRCLE || kind_ == TASK_TWO_CIRCLE) &&
			bb->LockedManeuverTask != static_cast<int>(kind_))
		{
			bb->LockedManeuverTask = static_cast<int>(kind_);
			bb->ManeuverTurnDegrees = 0.0f;
			bb->ManeuverStartTime = static_cast<float>(bb->RunningTime);
			const double hostileTurnSide = targetTurnSide(bb);
			// Paper/BEM top-view definition: one-circle uses opposite world
			// turn directions; two-circle uses the same world turn direction.
			bb->LockedManeuverSide = static_cast<float>(
				kind_ == TASK_ONE_CIRCLE ? -hostileTurnSide : hostileTurnSide);
			bb->PreviousManeuverForward = levelForward(bb);
		}
		if (kind_ == TASK_SCISSORS &&
			bb->LockedManeuverTask != static_cast<int>(TASK_SCISSORS))
		{
			startManeuverLock(bb, static_cast<int>(TASK_SCISSORS), turnSide(bb));
		}
		const bool lockedTurnTask =
			commandKind == TASK_ONE_CIRCLE ||
			commandKind == TASK_TWO_CIRCLE ||
			commandKind == TASK_SCISSORS;
		const double side = lockedTurnTask && std::abs(bb->LockedManeuverSide) > 0.5f
			? bb->LockedManeuverSide
			: turnSide(bb);
		const BT_Geometry::Vector3 turnForward = levelForward(bb);
		const BT_Geometry::Vector3 turnRight = levelRight(bb);
		const BT_Geometry::Vector3 worldUp(0, 0, 1);
		const double corner = maneuverSpeed(bb);
		const BT_Geometry::Vector3 pure = bb->TargetLocaion_Cartesian;
		const BT_Geometry::Vector3 lead = bb->PredictedTargetLocation;
		// Lag offset scales with range: a fixed 1200 m offset at sub-1000 m
		// range puts the lag point behind US, so the aim diverges as we close
		// (observed: Track stalled at ~12 deg LOS and fell back out).
		const double lagDistance = clampValue(0.4 * bb->Distance, 150.0, 1200.0);
		const BT_Geometry::Vector3 lag = bb->TargetLocaion_Cartesian - bb->TargetForwardVector * lagDistance;
		BT_Geometry::Vector3 vp = pure;
		ThrottleMode mode = THR_AUTO;
		double speed = corner;
		// PN gating: raised only when the VP rides the target's motion, so the
		// acceleration controller may use the analytic LOS rate. The anchor is
		// the target-tied base point of the VP, EXCLUDING own-attitude terms
		// (alpha bias, body-frame offsets) so PN never chases our own dynamics.
		// Its velocity comes from a finite difference, valid only while the
		// same task stays active (jump guard rejects branch-toggle spikes).
		BT_Geometry::Vector3 vpVelocity(0, 0, 0);
		BT_Geometry::Vector3 pnAnchor = pure;
		bool vpTied = false;
		switch (commandKind)
		{
		case TASK_CLIMB_RECOVER:
			vp = groundRecoveryVP(bb);
			mode = THR_MAX;
			break;
		case TASK_BREAK_JINK:
		{
			// DBFM initial break: first leave the hostile cone, then add a small
			// vertical jink. A pure time-based left/right reversal can pick the
			// wrong side for one scoring tick, which is enough to eat phase-1
			// damage in the public cone rule.
			const double defensiveSide = committedDefensiveSide(bb, number("JinkCommitSec", 1.05));
			const double period = std::max(number("JinkPeriodSec", 0.9), 0.2);
			const double phase = bb->RunningTime * TWO_PI / period;
			const double rangeFt = bb->Distance / FT;
			const bool enemyActualCone =
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) > 0.0;
			const bool reformingGunThreat =
				bb->BFM == DBFM &&
				rangeFt < 2600.0 &&
				bb->Los_Degree > 95.0f &&
				bb->Los_Degree_Target < std::max(phaseConeDeg(bb->Phase) + 26.0, 28.0) &&
				bb->EnemyLosRate_DegSec < -8.0f;
			const bool emergencyConeEscape =
				enemyActualCone ||
				reformingGunThreat ||
				(bb->Los_Degree_Target < phaseConeDeg(bb->Phase) + 2.0 &&
				 rangeFt < phaseMaxRangeFt(bb->Phase) + 700.0);
			if (emergencyConeEscape)
			{
				// Centered gun threats still need an immediate out-of-plane break.
				// Near the cone edge, add lateral displacement so DBFM jinking is
				// not reduced to a pitch-only pull.
				const double coneDeg = phaseConeDeg(bb->Phase);
				const bool nearConeEdge =
					enemyActualCone &&
					bb->Los_Degree_Target > std::max(0.55 * coneDeg, coneDeg - 0.35) &&
					rangeFt > 1200.0;
				if (nearConeEdge)
					vp = enemyConeEscapeVP(bb, defensiveSide, 9000.0, 800.0, 1600.0);
				else
					vp = enemyConeEscapeVP(bb, defensiveSide, 5200.0, 800.0, 5200.0);
			}
			else
			{
				vp = enemyConeEscapeVP(bb, defensiveSide, 10500.0, 650.0,
					700.0 + std::cos(phase) * 350.0);
			}
			// BEM DBFM flow explicitly commands PWR MAX after the break.
			mode = THR_MAX;
			speed = std::max(corner, static_cast<double>(bb->MySpeed_MS) + 20.0);
			break;
		}
		case TASK_HARD_TURN:
		{
			const double defensiveSide = committedDefensiveSide(bb, 2.5);
			const double rangeFt = bb->Distance / FT;
			const double closingFtps = std::max(0.0, -static_cast<double>(bb->ClosureRate_MS) / FT);
			const bool enemyActualCone =
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) > 0.0;
			const bool pureOrLeadRearThreat =
				bb->BFM == DBFM &&
				(bb->EnemyPursuitType == EPT_PURE ||
				 bb->EnemyPursuitType == EPT_LEAD ||
				 bb->Los_Degree_Target < phaseConeDeg(bb->Phase) + 6.0) &&
				bb->Los_Degree > 80.0f &&
				rangeFt < 6800.0;
			if (enemyActualCone || pureOrLeadRearThreat)
			{
				// VER17: HardTurn was safe but sterile in node-effect probes.
				// Keep the same DBFM node, but make the initial turn solve the
				// measured problem: increase enemyLOS quickly and stop holding a
				// planar orbit after the cone is defeated.
				const double lateral = enemyActualCone ? 700.0 : 9000.0;
				const bool roomyPredictedThreat =
					!enemyActualCone &&
					rangeFt > 2600.0 &&
					bb->Los_Degree_Target > phaseConeDeg(bb->Phase) + 0.2f &&
					bb->Los_Degree_Target < phaseConeDeg(bb->Phase) + 9.0f;
				const double vertical = enemyActualCone ? 9800.0 :
					roomyPredictedThreat
					// Low-closure roomy threats can drift back into the cone if
					// the break is nearly level. Keep high-closure threats mostly
					// lateral, but retain a bounded out-of-plane pull when there
					// is enough room to deny the tail shot before it matures.
					? (closingFtps > 350.0 ? 250.0 : 900.0)
					: (closingFtps > 350.0 ? 1700.0 : 1100.0);
				vp = enemyConeEscapeVP(bb, defensiveSide, lateral, 850.0, vertical);
			}
			else
				vp = bb->MyLocation_Cartesian + turnRight * (defensiveSide * 6000.0) + turnForward * 500.0;
			mode = THR_MAX;
			speed = std::max(corner, static_cast<double>(bb->MySpeed_MS) + 12.0);
			break;
		}
		case TASK_CONE_LEAD_TRACK:
		{
			bb->TrackSubMode = TRACK_LEAD;
			bb->TrackSubReason = TRACK_REASON_TERMINAL_LEAD;
			bb->VPPMode = VPP_LEAD;
			bb->ShotCommitUntil = static_cast<float>(std::max(static_cast<double>(bb->ShotCommitUntil),
				bb->RunningTime + number("ShotCommitSec", 0.75)));
			const double rangeFt = bb->Distance / FT;
			const double rate = apparentLosRateDegSec(bb);
			const double closingFtps = std::max(0.0, -static_cast<double>(bb->ClosureRate_MS) / FT);
			const double scoringRangeFt = phaseMaxRangeFt(bb->Phase);
			const bool lowAspectHighClosureTerminal =
				bb->Phase <= 1 &&
				rangeFt > 700.0 &&
				rangeFt < scoringRangeFt + 1200.0 &&
				closingFtps > 750.0 &&
				bb->Los_Degree > 3.5f &&
				bb->Los_Degree_Target < 15.0f &&
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0;
			const double baseLead = number("LeadPredictionSec", 0.30);
			const double tLead = clampValue(
				baseLead + clampValue(rate / 90.0, 0.0, lowAspectHighClosureTerminal ? 0.30 : 0.20),
				lowAspectHighClosureTerminal ? 0.20 : 0.12,
				lowAspectHighClosureTerminal ? 0.95 : 0.60);
			const bool phaseOneTerminal = bb->Phase <= 1 &&
				rangeFt < scoringRangeFt + (lowAspectHighClosureTerminal ? 1200.0 : 700.0) &&
				bb->Los_Degree < (lowAspectHighClosureTerminal ? 22.0f : 18.0f);
			const bool phaseOneRearCenterAim =
				phaseOneTerminal &&
				bb->Los_Degree < 15.0f &&
				bb->Los_Degree_Target > 20.0f &&
				rangeFt > 700.0 &&
				rangeFt < scoringRangeFt + 700.0 &&
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0;
			const double leadLimitDeg = phaseOneTerminal
				? (lowAspectHighClosureTerminal
					? std::min(number("MaxLeadDeg", 24.0), 24.0)
					: std::min(number("MaxLeadDeg", 12.0), 8.0))
				: number("MaxLeadDeg", 12.0);
			BT_Geometry::Vector3 leadPoint = angularlyLimitedLeadAim(bb,
				predictedTargetAt(bb, tLead), leadLimitDeg);
			const double leadWeight = lowAspectHighClosureTerminal
				? clampValue((closingFtps - 650.0) / 850.0, 0.55, 0.85)
				: phaseOneTerminal
				? clampValue((rate - 2.0) / 18.0, 0.20, 0.65)
				: clampValue((rate - 2.0) / 18.0, 0.35, 0.90);
			vp = pure * (1.0 - leadWeight) + leadPoint * leadWeight;
			if (phaseOneRearCenterAim)
			{
				// The contest cone is fixed to the nose axis.  Once OBFM has
				// achieved a rear-aspect phase-1 near-cone, do not let a lead VP
				// pull the nose away from the target center; use the same Track
				// node as the paper Employ/Track terminal handoff, but make the
				// aim point body-axis compatible with the simplified cone rule.
				vp = pure;
				pnAnchor = pure;
				bb->TrackSubMode = TRACK_PURE;
				bb->TrackSubReason = TRACK_REASON_PHASE1_FINE_PURE;
				bb->VPPMode = VPP_PURE;
			}
			else if (lowAspectHighClosureTerminal && bb->Los_Degree > 1.4f)
			{
				vp = snapThroughGunAim(bb, vp, 2.65, 9.0);
				bb->TrackSubMode = TRACK_SNAP;
				bb->TrackSubReason = TRACK_REASON_PHASE1_LEAD_SNAP;
			}
			else if (phaseOneTerminal && closingFtps > 650.0 && bb->Los_Degree > 2.0f)
			{
				vp = snapThroughGunAim(bb, vp, 1.85, 7.0);
				bb->TrackSubMode = TRACK_SNAP;
				bb->TrackSubReason = TRACK_REASON_PHASE1_LEAD_SNAP;
			}
			else if (phaseOneTerminal && bb->Los_Degree < 1.6f)
			{
				vp = pure;
				bb->TrackSubMode = TRACK_PURE;
				bb->TrackSubReason = TRACK_REASON_PHASE1_FINE_PURE;
				bb->VPPMode = VPP_PURE;
			}
			pnAnchor = vp;
			Optional<bool> alphaBias = getInput<bool>("AlphaBias");
			const bool adaptiveAlphaBias =
				bb->MyAOA_Degree > 14.0f &&
				bb->Los_Degree > 3.5f &&
				rangeFt < phaseMaxRangeFt(bb->Phase) + 1800.0 &&
				closingFtps > 500.0;
			if ((alphaBias && alphaBias.value()) || adaptiveAlphaBias)
			{
				// Track/ConeLeadTrack use the nose-pointing terminal controller
				// in CPPBehaviorTree.  Alpha bias belongs to the velocity-pointing
				// acceleration controller; applying it here pushes the damage cone
				// away from the target in the final tracking segment.
				vp = alphaBiasedAim(bb, vp, bb->MyRightVector, false);
				if (bb->AlphaBiasApplied)
				{
					bb->TrackSubMode = TRACK_ALPHA;
					bb->TrackSubReason = TRACK_REASON_ALPHA;
				}
			}

			const double desiredRange = number("HoldRangeFt", 1700.0) * FT;
			const bool strictCone = damageRate(bb->Los_Degree, rangeFt, bb->Phase) > 0.0;
			const double baseSpeed = bb->TargetSpeed_MS + 0.06 * (bb->Distance - desiredRange) +
				0.08 * bb->ClosureRate_MS;
			const double minAimSpeed = strictCone && rangeFt < 2200.0
				? std::max(145.0, static_cast<double>(bb->TargetSpeed_MS))
				: std::max(0.88 * corner, 165.0);
			speed = clampValue(std::max(baseSpeed, minAimSpeed), 145.0, 300.0);

			// VER08: prevent the repeated 1-45 deg pass-through seen in seed 5.
			// When the target is already near the phase band and closure is high,
			// bleed energy instead of maintaining a 450-500 KCAS crossing.
			if (!strictCone &&
				rangeFt < phaseMaxRangeFt(bb->Phase) + 1800.0 &&
				closingFtps > 450.0 &&
				bb->MyKCAS_KT > (lowAspectHighClosureTerminal ? 0.90 : 0.72) * IDENTIFIED_CORNER_KCAS)
			{
				mode = THR_IDLE;
				speed = std::max(135.0, static_cast<double>(bb->TargetSpeed_MS) - 30.0);
				bb->ThrottleReason = THROTTLE_REASON_CLOSE_BRAKE;
			}
			else if (!strictCone &&
				(rangeFt > phaseMaxRangeFt(bb->Phase) + 500.0 ||
				 bb->MyKCAS_KT < 0.82 * IDENTIFIED_CORNER_KCAS))
			{
				mode = THR_MAX;
				bb->ThrottleReason = THROTTLE_REASON_REACCEL;
			}
			if (!strictCone &&
				rangeFt < phaseMaxRangeFt(bb->Phase) + 1800.0 &&
				closingFtps > 420.0 &&
				bb->MyKCAS_KT < 265.0f &&
				bb->Los_Degree < 5.0f)
			{
				vp = snapThroughGunAim(bb, vp, 1.65, 6.0);
				pnAnchor = vp;
				bb->TrackSubReason = TRACK_REASON_LOW_SPEED_SNAP;
			}
			vpTied = true;
			break;
		}
		case TASK_TRACK:
		{
			bb->ShotCommitUntil = static_cast<float>(std::max(static_cast<double>(bb->ShotCommitUntil),
				bb->RunningTime + number("ShotCommitSec", 0.75)));
			const double blend = clampValue(bb->Los_Degree / number("LagBlendDeg", 9999.0), 0.0, 1.0);
			bb->TrackSubMode = TRACK_BLEND;
			bb->TrackSubReason = TRACK_REASON_BLEND;
			bb->VPPMode = blend < 0.10 ? VPP_PURE : VPP_BLEND;
			vp = pure * (1.0 - blend) + lag * blend;
			pnAnchor = vp;	// pre-alpha-bias blend: rides the target only
			const double rangeFt = bb->Distance / FT;
			const double closingFtps = std::max(0.0, -static_cast<double>(bb->ClosureRate_MS) / FT);
			Optional<bool> alphaBias = getInput<bool>("AlphaBias");
			const bool adaptiveAlphaBias =
				bb->MyAOA_Degree > 14.0f &&
				bb->Los_Degree > 3.5f &&
				rangeFt < phaseMaxRangeFt(bb->Phase) + 1800.0 &&
				closingFtps > 500.0;
			if ((alphaBias && alphaBias.value()) || adaptiveAlphaBias)
			{
				// Track/ConeLeadTrack are terminal nose-pointing tasks, so velocity
				// alpha compensation is intentionally disabled here.
				vp = alphaBiasedAim(bb, vp, bb->MyRightVector, false);
				if (bb->AlphaBiasApplied)
				{
					bb->TrackSubMode = TRACK_ALPHA;
					bb->TrackSubReason = TRACK_REASON_ALPHA;
				}
			}
			const bool strictCone = damageRate(bb->Los_Degree, rangeFt, bb->Phase) > 0.0;
			const double coneDeg = phaseConeDeg(bb->Phase);
			const double targetVerticalErrorDeg = signedVerticalErrorDeg(
				bb->MyForwardVector, bb->MyUpVector, pure);
			const bool phaseOneHighClosureVerticalTrack =
				!strictCone &&
				bb->Phase <= 1 &&
				rangeFt > 650.0 &&
				rangeFt < phaseMaxRangeFt(bb->Phase) + 650.0 &&
				closingFtps > 850.0 &&
				bb->Los_Degree > coneDeg + 0.15f &&
				bb->Los_Degree < coneDeg + 2.2f &&
				bb->Los_Degree_Target > coneDeg + 5.0f &&
				std::abs(targetVerticalErrorDeg) > 0.8 &&
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0;
			const bool phaseOneFinePureTrack =
				!strictCone &&
				bb->Phase <= 1 &&
				rangeFt > 600.0 &&
				rangeFt < phaseMaxRangeFt(bb->Phase) + 450.0 &&
				bb->Los_Degree > coneDeg + 1.0f &&
				bb->Los_Degree < 10.0f &&
				bb->Los_Degree_Target > coneDeg + 5.0f &&
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0;
			Optional<bool> coneCenterSnap = getInput<bool>("ConeCenterSnap");
			if (coneCenterSnap && coneCenterSnap.value() &&
				!strictCone &&
				bb->Phase <= 1 &&
				rangeFt > 800.0 &&
				rangeFt < phaseMaxRangeFt(bb->Phase) + 700.0 &&
				closingFtps > 500.0 &&
				bb->Los_Degree > 8.0f &&
				bb->Los_Degree < 75.0f &&
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0)
			{
				// Optional close-bridge behavior: the public phase-1 cone is much
				// stricter than the paper WEZ cue.  When this node is explicitly
				// selected as a terminal bridge, command through the pure cone
				// center instead of merely chasing the current LOS.
				vp = snapThroughGunAim(bb, pure, number("SnapGain", 2.05),
					number("SnapExtraDeg", 16.0));
				pnAnchor = vp;
				bb->TrackSubMode = TRACK_SNAP;
				bb->TrackSubReason = TRACK_REASON_CONE_CENTER_SNAP;
			}
			const bool preRangePureHold =
				!strictCone &&
				bb->Phase <= 1 &&
				rangeFt > phaseMaxRangeFt(bb->Phase) &&
				rangeFt < phaseMaxRangeFt(bb->Phase) + 900.0 &&
				bb->Los_Degree < 2.0f &&
				bb->Los_Degree_Target > 20.0f;
			if (preRangePureHold)
			{
				// Paper VPP smoothing passes through pure pursuit before lead.
				// In phase 1 this prevents a near-perfect pre-range cone from
				// being pulled back into lag/lead just before the 3000 ft gate.
				vp = pure;
				pnAnchor = vp;
				bb->TrackSubMode = TRACK_PURE;
				bb->TrackSubReason = TRACK_REASON_PHASE1_PRESCORE;
				bb->VPPMode = VPP_PURE;
			}
			if (phaseOneFinePureTrack)
			{
				// In phase 1 the contest cone is the nose axis itself, not a
				// projectile-lead solution. Once OBFM/Track has already driven
				// LOS below 8 deg inside 3000 ft, keep the paper Employ node on
				// pure target-center instead of re-opening the cone with a
				// contest-specific snap-through aim point.
				const bool lowEnergyReopeningFineTrack =
					bb->MyKCAS_KT < 230.0f &&
					closingFtps > 650.0 &&
					bb->MyLosRate_DegSec > 0.5f;
				vp = lowEnergyReopeningFineTrack
					? snapThroughGunAim(bb, pure, 2.55, 8.0)
					: pure;
				pnAnchor = vp;
				bb->TrackSubMode = lowEnergyReopeningFineTrack ? TRACK_SNAP : TRACK_PURE;
				bb->TrackSubReason = lowEnergyReopeningFineTrack
					? TRACK_REASON_LOW_SPEED_SNAP
					: TRACK_REASON_PHASE1_FINE_PURE;
				bb->VPPMode = VPP_PURE;
			}
			if (strictCone &&
				bb->Phase <= 1 &&
				rangeFt > 700.0 &&
				rangeFt < phaseMaxRangeFt(bb->Phase) + 120.0 &&
				closingFtps > 650.0 &&
				bb->Los_Degree < coneDeg + 0.40f &&
				bb->MyLosRate_DegSec > -1.5f &&
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0)
			{
				// The competition phase-1 cone is so narrow that a centered,
				// high-closure shot can still be lost in the next few ticks.
				// Keep the paper Employ node active but bias through centerline
				// when LOS has stopped improving inside the strict cone.
				const double snapScale = bb->MyLosRate_DegSec > 5.0f
					? 2.75
					: (bb->MyLosRate_DegSec > 0.5f ? 2.15 : 1.45);
				const double snapLimit = bb->MyLosRate_DegSec > 5.0f
					? 8.0
					: (bb->MyLosRate_DegSec > 0.5f ? 6.0 : 3.0);
				vp = snapThroughGunAim(bb, pure, snapScale, snapLimit);
				pnAnchor = vp;
				bb->TrackSubMode = TRACK_SNAP;
				bb->TrackSubReason = TRACK_REASON_STRICT_CONE;
			}
			if (!phaseOneFinePureTrack && !strictCone &&
				rangeFt < phaseMaxRangeFt(bb->Phase) + 1800.0 &&
				closingFtps > 420.0 &&
				bb->Los_Degree > std::max(4.0, coneDeg + 2.0) &&
				bb->Los_Degree < std::max(28.0, coneDeg + 9.0))
			{
				// Paper-equivalent terminal LeadVPP: once the control-zone flow
				// has reduced LOS but closure is still high, do not pure-pursue
				// the current target point. Aim a short horizon ahead so the
				// nose cone meets the target during the pass.
				const double leadSec = clampValue(
					number("LeadPredictionSec", 0.14) + clampValue(closingFtps / 3200.0, 0.0, 0.20),
					0.10, 0.38);
				const double leadWeight = clampValue((closingFtps - 420.0) / 900.0, 0.18, 0.70);
				const BT_Geometry::Vector3 leadAim = angularlyLimitedLeadAim(
					bb, predictedTargetAt(bb, leadSec), number("MaxLeadDeg", 14.0));
				vp = vp * (1.0 - leadWeight) + leadAim * leadWeight;
				pnAnchor = vp;
				bb->TrackSubMode = TRACK_LEAD;
				bb->TrackSubReason = TRACK_REASON_TERMINAL_LEAD;
				bb->VPPMode = VPP_LEAD;
			}
			if (!phaseOneFinePureTrack && !strictCone &&
				bb->Phase <= 1 &&
				rangeFt > 1500.0 &&
				rangeFt < phaseMaxRangeFt(bb->Phase) + 700.0 &&
				closingFtps > 650.0 &&
				bb->Los_Degree > 18.0f &&
				bb->Los_Degree < 32.0f &&
				bb->Los_Degree_Target > 6.0f &&
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0)
			{
				const double leadSec = clampValue(0.18 + closingFtps / 5200.0, 0.22, 0.42);
				const BT_Geometry::Vector3 leadAim = angularlyLimitedLeadAim(
					bb, predictedTargetAt(bb, leadSec), 18.0);
				vp = snapThroughGunAim(bb, leadAim, 3.25, 20.0);
				pnAnchor = vp;
				bb->TrackSubMode = TRACK_SNAP;
				bb->TrackSubReason = TRACK_REASON_PHASE1_LEAD_SNAP;
				bb->VPPMode = VPP_LEAD;
			}
			if (!phaseOneFinePureTrack && !strictCone &&
				rangeFt < phaseMaxRangeFt(bb->Phase) + 1800.0 &&
				closingFtps > 420.0 &&
				bb->MyKCAS_KT < 300.0f &&
				bb->Los_Degree < 5.0f)
			{
				const bool rearAspectTrack = bb->Los_Degree_Target > 25.0f;
				vp = rearAspectTrack
					? snapThroughGunAim(bb, vp, 2.20, 8.0)
					: snapThroughGunAim(bb, vp, 1.55, 5.0);
				pnAnchor = vp;
				bb->TrackSubMode = TRACK_SNAP;
				bb->TrackSubReason = TRACK_REASON_LOW_SPEED_SNAP;
			}
			if (!phaseOneFinePureTrack && !strictCone &&
				bb->Phase <= 1 &&
				rangeFt < phaseMaxRangeFt(bb->Phase) + 2400.0 &&
				bb->Los_Degree < 10.0f &&
				bb->Los_Degree_Target > 20.0f)
			{
				// The public contest scores phase 1 only inside 1 deg/3000 ft.
				// Once the paper flow has produced a near-cone rear aspect, the
				// terminal action must drive the nose through the damage cone.
				// If the target is well behind its 3/9 line and closure is high,
				// pure pursuit reaches a 2-5 deg miss before the scoring range and
				// then opens. Start the paper-equivalent terminal lead earlier and
				// keep a bounded snap-through command until the LOS is nearly in
				// the phase cone.
				const bool nearScoreGate =
					rangeFt < phaseMaxRangeFt(bb->Phase) + 450.0 &&
					bb->Los_Degree < 8.0f;
				const bool longPreScoreRearAspect =
					rangeFt > phaseMaxRangeFt(bb->Phase) + 1500.0 &&
					closingFtps > 750.0 &&
					bb->MyLosRate_DegSec > -2.0f;
				vp = (preRangePureHold || nearScoreGate)
					? pure
					: (longPreScoreRearAspect
						? snapThroughGunAim(bb, pure, 3.10, 14.0)
						: snapThroughGunAim(bb, pure, 2.35, 8.0));
				pnAnchor = vp;
				bb->TrackSubMode = (preRangePureHold || nearScoreGate) ? TRACK_PURE : TRACK_SNAP;
				bb->TrackSubReason = TRACK_REASON_PHASE1_PRESCORE;
				bb->VPPMode = VPP_PURE;
			}
			if (!phaseOneFinePureTrack && !strictCone &&
				bb->Phase <= 1 &&
				rangeFt < phaseMaxRangeFt(bb->Phase) + 700.0 &&
				rangeFt > 550.0 &&
				closingFtps > 500.0 &&
				bb->Los_Degree > 1.0f &&
				bb->Los_Degree < 18.0f &&
				bb->Los_Degree_Target > 6.0f &&
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0)
			{
				// Contest phase 1 is stricter than the paper's generic gun WEZ:
				// damage exists only inside 1 deg / 3000 ft.  The paper flow can
				// correctly deliver us to Track, yet still miss the scoring window
				// by 1-3 degrees during a high-closure pass.  In that final band,
				// keep the same Track node active but bias the aim point through
				// the cone center instead of accepting a pure-pursuit fly-through.
				const bool clearRearAspect = bb->Los_Degree_Target > 20.0f;
				const bool closeMidAspect = !clearRearAspect &&
					bb->Los_Degree_Target > 6.0f &&
					rangeFt < phaseMaxRangeFt(bb->Phase) + 700.0 &&
					bb->Los_Degree < 18.0f;
				const bool highClosureMidAspect = !clearRearAspect &&
					bb->Los_Degree_Target > 6.0f &&
					closingFtps > 850.0 &&
					rangeFt < phaseMaxRangeFt(bb->Phase) + 700.0;
				const bool lowEnergyFineTrack =
					bb->MyKCAS_KT < 245.0f &&
					bb->MyAOA_Degree > 15.0f &&
					bb->Los_Degree < 12.0f;
				const bool scoreGatePureCapture =
					rangeFt < phaseMaxRangeFt(bb->Phase) + 220.0 &&
					bb->MyLosRate_DegSec > -1.0f &&
					bb->Los_Degree < coneDeg + 0.35f;
				const bool openingNearGatePureCapture =
					rangeFt < phaseMaxRangeFt(bb->Phase) + 260.0 &&
					bb->MyLosRate_DegSec > 0.5f &&
					bb->Los_Degree < coneDeg + 0.30f &&
					bb->Los_Degree_Target > 20.0f;
				const bool clearRearReopeningSnap =
					clearRearAspect &&
					rangeFt < phaseMaxRangeFt(bb->Phase) + 250.0 &&
					bb->MyLosRate_DegSec > 0.5f &&
					bb->Los_Degree < 4.5f;
				const bool clearRearNearGatePure =
					clearRearAspect &&
					bb->Phase <= 1 &&
					rangeFt < phaseMaxRangeFt(bb->Phase) + 500.0 &&
					bb->Los_Degree < 4.5f;
				const bool closeMidNearGatePure =
					closeMidAspect &&
					bb->Phase <= 1 &&
					rangeFt < phaseMaxRangeFt(bb->Phase) + 500.0 &&
					bb->Los_Degree < coneDeg + 4.2f;
				const bool preScoreMidAspectSnap =
					closeMidNearGatePure &&
					rangeFt > phaseMaxRangeFt(bb->Phase) - 50.0 &&
					rangeFt < phaseMaxRangeFt(bb->Phase) + 550.0 &&
					bb->Los_Degree > coneDeg + 0.05f &&
					bb->MyLosRate_DegSec > -2.0f &&
					damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0;
				const bool rearAspectOpeningAfterShot =
					clearRearAspect &&
					bb->MyLosRate_DegSec > 0.5f &&
					bb->Los_Degree > coneDeg + 0.05f;
				if ((lowEnergyFineTrack && !clearRearAspect && !closeMidAspect && !clearRearReopeningSnap) ||
					(scoreGatePureCapture && !rearAspectOpeningAfterShot) ||
					(openingNearGatePureCapture && !rearAspectOpeningAfterShot) ||
					(clearRearNearGatePure && !rearAspectOpeningAfterShot) ||
					closeMidNearGatePure)
					vp = preScoreMidAspectSnap
						? snapThroughGunAim(bb, pure, 2.85, 8.0)
						: pure;
				else if (rearAspectOpeningAfterShot || clearRearReopeningSnap)
					vp = snapThroughGunAim(bb, pure, 3.10, 12.0);
				else
					vp = clearRearAspect
						? snapThroughGunAim(bb, pure, 2.60, 10.0)
						: (highClosureMidAspect
							? snapThroughGunAim(bb, pure, 3.10, 12.0)
							: (closeMidAspect
							? snapThroughGunAim(bb, pure, 3.20, 11.0)
								: snapThroughGunAim(bb, pure, 2.20, 6.0)));
				pnAnchor = vp;
				bb->TrackSubMode = (preRangePureHold || scoreGatePureCapture ||
					openingNearGatePureCapture || clearRearNearGatePure || closeMidNearGatePure) &&
					!preScoreMidAspectSnap && !rearAspectOpeningAfterShot && !clearRearReopeningSnap
					? TRACK_PURE : TRACK_SNAP;
				bb->TrackSubReason = TRACK_REASON_PHASE1_FINAL;
				bb->VPPMode = VPP_PURE;
			}
			if (phaseOneHighClosureVerticalTrack)
			{
				// High-closure phase-1 misses are usually not a pursuit-mode
				// problem: the paper Employ node is already active, but the
				// 1-degree contest cone is crossed with a small vertical error.
				// Keep the terminal action on target-center and apply only a
				// bounded vertical bias while throttle bleeds closure below.
				BT_Geometry::Vector3 upAim = bb->MyUpVector;
				upAim.normalize();
				const double biasDeg = clampValue(std::abs(targetVerticalErrorDeg) * 0.55, 0.25, 1.4);
				const double biasMeters = std::tan(biasDeg / DEG) * bb->Distance;
				vp = pure + upAim * (targetVerticalErrorDeg >= 0.0 ? biasMeters : -biasMeters);
				pnAnchor = pure;
				bb->TrackSubMode = TRACK_SNAP;
				bb->TrackSubReason = TRACK_REASON_HIGH_CLOSURE_VERTICAL;
				bb->VPPMode = VPP_PURE;
			}
			const double baseSpeed = bb->TargetSpeed_MS +
				0.06 * (bb->Distance - number("HoldRangeFt", 1700.0) * FT) +
				0.08 * bb->ClosureRate_MS;
			const double minAimSpeed = strictCone && rangeFt < 2200.0
				? std::max(145.0, static_cast<double>(bb->TargetSpeed_MS))
				: std::max(0.88 * corner, 165.0);
			speed = clampValue(std::max(baseSpeed, minAimSpeed), 145.0, 300.0);
			const bool phaseOneTerminalFineAim =
				!strictCone &&
				bb->Phase <= 1 &&
				rangeFt < phaseMaxRangeFt(bb->Phase) + 2200.0 &&
				rangeFt > 700.0 &&
				closingFtps > 500.0 &&
				bb->Los_Degree < 22.0f;
			const bool preRangeNearConeBrake =
				!strictCone &&
				bb->Phase <= 1 &&
				rangeFt > phaseMaxRangeFt(bb->Phase) + 260.0 &&
				rangeFt < phaseMaxRangeFt(bb->Phase) + 1400.0 &&
				closingFtps > 750.0 &&
				bb->Los_Degree < 7.0f &&
				bb->Los_Degree_Target > 20.0f;
			const bool nearGateClosureBrake =
				!strictCone &&
				bb->Phase <= 1 &&
				rangeFt < phaseMaxRangeFt(bb->Phase) + 650.0 &&
				rangeFt > phaseMaxRangeFt(bb->Phase) - 250.0 &&
				closingFtps > 750.0 &&
				bb->Los_Degree < 8.0f &&
				bb->Los_Degree_Target > 20.0f;
			const bool preserveTerminalCorner =
				!strictCone &&
				bb->Phase <= 1 &&
				rangeFt > phaseMaxRangeFt(bb->Phase) - 100.0 &&
				rangeFt < phaseMaxRangeFt(bb->Phase) + 2300.0 &&
				closingFtps > 500.0 &&
				bb->Los_Degree > 12.0f &&
				bb->Los_Degree < 26.0f &&
				(bb->Los_Degree_Target - bb->Los_Degree > 6.0f ||
				 (bb->Los_Degree_Target > 6.0f &&
				  bb->Los_Degree_Target - bb->Los_Degree > -28.0f &&
				  damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0)) &&
				bb->MyKCAS_KT < 0.90 * IDENTIFIED_CORNER_KCAS;
			const bool fineTrackLowEnergy =
				phaseOneFinePureTrack &&
				(bb->MyKCAS_KT < 230.0f || bb->MyAOA_Degree > 15.0f);
			const bool fineTrackHighClosureBrake =
				(phaseOneFinePureTrack || phaseOneHighClosureVerticalTrack) &&
				!fineTrackLowEnergy &&
				closingFtps > 700.0;
			if (fineTrackHighClosureBrake)
			{
				mode = THR_IDLE;
				speed = std::max(135.0, static_cast<double>(bb->TargetSpeed_MS) - 35.0);
				bb->ThrottleReason = THROTTLE_REASON_TERMINAL_BRAKE;
				bb->TrackSubReason = TRACK_REASON_HIGH_CLOSURE_VERTICAL;
			}
			else if (fineTrackLowEnergy)
			{
				// A low-energy Track node should not keep asking the nose to do
				// fine cone work while the aircraft decays below maneuver speed.
				// Preserve pure target-center aiming, but recover toward corner
				// instead of coasting in THR_AUTO.
				mode = THR_MAX;
				speed = std::max(speed, std::max(0.98 * corner, static_cast<double>(bb->TargetSpeed_MS) + 20.0));
				bb->ThrottleReason = THROTTLE_REASON_FINE_TRACK_ENERGY;
				bb->TrackSubReason = TRACK_REASON_LOW_ENERGY_FINE;
			}
			else if (preRangeNearConeBrake || nearGateClosureBrake)
			{
				mode = THR_IDLE;
				speed = std::max(135.0, static_cast<double>(bb->TargetSpeed_MS) - 45.0);
				bb->ThrottleReason = THROTTLE_REASON_TERMINAL_BRAKE;
			}
			else if (preserveTerminalCorner)
			{
				mode = THR_MAX;
				speed = std::max(speed, std::max(0.98 * corner, static_cast<double>(bb->TargetSpeed_MS) + 20.0));
				bb->ThrottleReason = THROTTLE_REASON_TERMINAL_CORNER;
			}
			else if (phaseOneTerminalFineAim && bb->MyKCAS_KT < 315.0f)
			{
				mode = THR_MAX;
				speed = std::max(speed, std::max(0.98 * corner, static_cast<double>(bb->TargetSpeed_MS) + 20.0));
				bb->ThrottleReason = THROTTLE_REASON_FINE_TRACK_ENERGY;
			}
			else if (!strictCone && rangeFt < phaseMaxRangeFt(bb->Phase) + 2200.0 && closingFtps > 450.0)
			{
				mode = THR_IDLE;
				speed = std::max(135.0, static_cast<double>(bb->TargetSpeed_MS) - 35.0);
				bb->ThrottleReason = THROTTLE_REASON_CLOSE_BRAKE;
			}
			else if (!strictCone &&
				(rangeFt > phaseMaxRangeFt(bb->Phase) + 500.0 ||
				 bb->MyKCAS_KT < 0.82 * IDENTIFIED_CORNER_KCAS))
			{
				mode = THR_MAX;
				bb->ThrottleReason = THROTTLE_REASON_REACCEL;
			}
			vpTied = true;
			break;
		}
		case TASK_FOLLOW_HOSTILE:
		{
			// BEM block 7/10: follow hostile after entering the turn circle,
			// using You/Shim VPP smoothing rather than a fixed rear-target lag
			// point. The lag endpoint is turn-center/energy based; the weight
			// moves only through lag-pure-lead, preserving the paper's pursuit
			// transition order while adapting the probabilities to available
			// competition state.
			const double rangeFt = bb->Distance / FT;
			const double closingFtps = std::max(0.0, -static_cast<double>(bb->ClosureRate_MS) / FT);
			const double openingFtps = std::max(0.0, static_cast<double>(bb->ClosureRate_MS) / FT);
			const BT_Geometry::Vector3 lagVpp = virtualLagVPP(bb, corner, side, 0.75, 1800.0);
			double desiredVppWeight = 0.50;
			if (hostileVerticalUp(bb))
			{
				// High yo-yo style lag: preserve turn-circle position while
				// the vertical LagVPP bleeds surplus energy above the bandit.
				desiredVppWeight = 0.20;
				vp = smoothedVPP(bb, lagVpp, pure, lead, desiredVppWeight);
				mode = bb->MyKCAS_KT > 390.0f ? THR_IDLE : THR_MAX;
			}
			else if (hostileVerticalDown(bb))
			{
				// Do not dive through the hard-deck guard just because the
				// hostile points down; move from lag toward pure and keep an
				// altitude cushion.
				desiredVppWeight = clampValue(0.35 + bb->Los_Degree / 160.0, 0.35, 0.65);
				vp = smoothedVPP(bb, lagVpp + worldUp * 400.0, pure + worldUp * 300.0, lead, desiredVppWeight);
				mode = THR_AUTO;
			}
			else if (bb->Phase <= 1 &&
				rangeFt < phaseMaxRangeFt(bb->Phase) + 3000.0 &&
				rangeFt > phaseMaxRangeFt(bb->Phase) + 1200.0 &&
				bb->Los_Degree < 12.0f &&
				bb->Los_Degree_Target > 20.0f &&
				closingFtps > 500.0)
			{
				// You/Shim VPP smoothing assigns high pure/lead probability
				// inside the weapon geometry.  Do not let high closure alone
				// pull the VPP back to a far LagVPP once the nose is already
				// near the cone; that releases the pull just before Track.
				const double leadSec = clampValue(
					(rangeFt - phaseMaxRangeFt(bb->Phase) + 250.0) /
					std::max(closingFtps, 1.0),
					0.16, 0.40);
				const BT_Geometry::Vector3 leadAim = angularlyLimitedLeadAim(
					bb, predictedTargetAt(bb, leadSec), 5.5);
				desiredVppWeight = clampValue(
					0.58 + clampValue((12.0 - bb->Los_Degree) / 20.0, 0.0, 0.18),
					0.58, 0.76);
				bb->VppBlendWeight = static_cast<float>(
					std::max(static_cast<double>(bb->VppBlendWeight), 0.50));
				vp = smoothedVPP(bb, lagVpp, pure, leadAim, desiredVppWeight);
				mode = bb->MyKCAS_KT < 0.92 * IDENTIFIED_CORNER_KCAS ? THR_MAX : THR_AUTO;
			}
			else
			{
				// High closing rate or opening range asks for lag/control-zone
				// retention. Stable closure and small LOS move toward lead/gun.
				desiredVppWeight = 0.50 +
					clampValue((bb->Los_Degree_Target - 90.0) / 120.0, -0.20, 0.20) -
					clampValue(closingFtps / 1600.0, 0.0, 0.35) +
					clampValue(openingFtps / 2400.0, 0.0, 0.20);
				if (rangeFt < number("HoldRangeFt", 2500.0) + 400.0)
					desiredVppWeight -= clampValue(openingFtps / 2400.0, 0.0, 0.20);
				if (bb->Los_Degree < 3.0 * bb->Phase && bb->Distance / FT < 4200.0 && closingFtps < 350.0)
					desiredVppWeight = 0.85;
				desiredVppWeight = clampValue(desiredVppWeight, 0.05, 0.90);
				vp = smoothedVPP(bb, lagVpp, pure, lead, desiredVppWeight);
			}
			pnAnchor = vp;
			if (bb->Los_Degree < 2.0 * bb->Phase)
				vp = alphaBiasedAim(bb, vp, bb->MyRightVector);
			const double desiredRange = number("HoldRangeFt", 2500.0) * FT;
			speed = clampValue(
				std::max(0.94 * corner, bb->TargetSpeed_MS + 0.05 * (bb->Distance - desiredRange) + 0.10 * bb->ClosureRate_MS),
				150.0, 330.0);
			if (bb->Distance / FT > 5000.0 || (bb->Los_Degree > 35.0f && bb->MyKCAS_KT < 0.90 * IDENTIFIED_CORNER_KCAS))
				mode = THR_MAX;
			vpTied = true;
			break;
		}
		case TASK_HIGH_YOYO:
		{
			// Vertical lag, energy-sized: dz = (V^2 - Vc^2)/2g (spec eq. 17/19).
			const double excessHeight = (bb->MySpeed_MS * bb->MySpeed_MS - corner * corner) / (2.0 * G);
			const double climb = excessHeight > 120.0 && bb->MyKCAS_KT > 0.95 * IDENTIFIED_CORNER_KCAS
				? clampValue(excessHeight, 0.0, 1800.0)
				: 0.0;
			vp = lag + BT_Geometry::Vector3(0, 0, climb);
			pnAnchor = vp;
			// A high yo-yo spends surplus energy; below corner it must not keep
			// unloading energy, or the follow-on Track/Pull-to-HUD arrives slow
			// and saturated before the cone can close.
			mode = bb->MyKCAS_KT < 0.92 * IDENTIFIED_CORNER_KCAS ? THR_MAX : THR_IDLE;
			speed = std::max(corner, bb->TargetSpeed_MS + 12.0);
			vpTied = true;
			break;
		}
		case TASK_LEAD_INTERCEPT:
		{
			const double rangeFt = bb->Distance / FT;
			const double speedRatio = bb->MyKCAS_KT > 1.0f
				? static_cast<double>(bb->MyKCAS_KT) / IDENTIFIED_CORNER_KCAS
				: bb->MySpeed_MS / std::max(corner, 1.0);
			const BT_Geometry::Vector3 predicted =
				angularlyLimitedLeadAim(bb, predictedTargetAt(bb, 0.35), 8.0);
			if (bb->Los_Degree > 65.0f || rangeFt > 6500.0)
			{
				const double rejoinSide = committedRejoinSide(bb, predicted, 1, 5.0);
				const double maxTurnDeg = speedRatio < 0.70 ? 42.0 :
					(speedRatio < 0.90 ? 60.0 : 75.0);
				vp = horizontalRejoinVP(bb, predicted, maxTurnDeg, 5000.0, 180.0, rejoinSide);
				mode = THR_MAX;
				speed = std::max(corner, bb->TargetSpeed_MS + 20.0);
				vpTied = false;
			}
			else
			{
				vp = predicted;
				pnAnchor = predicted;
				speed = std::max(0.94 * corner, bb->TargetSpeed_MS + 12.0);
				mode = bb->MyKCAS_KT < 0.90 * IDENTIFIED_CORNER_KCAS ? THR_MAX : THR_AUTO;
				vpTied = true;
			}
			break;
		}
		case TASK_LAG_ENTRY:
		{
			// BEM block 6 enters the hostile turn circle tangentially. Select
			// the true tangent point from ownship that lies closest to the
			// requested behind-hostile entry window.
			BT_Geometry::Vector3 radial = bb->TargetLocaion_Cartesian - bb->TargetTurnCenter;
			BT_Geometry::Vector3 axis = radial.cross(bb->TargetVelocity);
			if (bb->TargetTurnCircleValid && axis.length() > 1.0)
			{
				axis.normalize();
				const double arcRad = number("EntryArcDeg", 40.0) / DEG;
				const BT_Geometry::Vector3 desiredEntry =
					bb->TargetTurnCenter + rotateAboutAxis(radial, axis, -arcRad);
				const BT_Geometry::Vector3 centerToOwn =
					bb->MyLocation_Cartesian - bb->TargetTurnCenter;
				BT_Geometry::Vector3 ownInTurnPlane =
					centerToOwn - axis * centerToOwn.dot(axis);
				const double planarDistance = ownInTurnPlane.length();
				const double radius = std::max(static_cast<double>(bb->TargetTurnRadius_M), 1.0);
				if (planarDistance > radius + 1.0)
				{
					const BT_Geometry::Vector3 ownCircleRadial = normalized(ownInTurnPlane) * radius;
					const double tangentAngle = std::acos(clampValue(radius / planarDistance, -1.0, 1.0));
					const BT_Geometry::Vector3 tangentA =
						bb->TargetTurnCenter + rotateAboutAxis(ownCircleRadial, axis, tangentAngle);
					const BT_Geometry::Vector3 tangentB =
						bb->TargetTurnCenter + rotateAboutAxis(ownCircleRadial, axis, -tangentAngle);
					vp = tangentA.distanceSquared(desiredEntry) <= tangentB.distanceSquared(desiredEntry)
						? tangentA : tangentB;
				}
				else
					vp = desiredEntry;
			}
			else
				vp = lag;
			pnAnchor = vp;
			mode = THR_MAX;
			vpTied = true;
			break;
		}
		case TASK_EXTEND:
			vp = bb->MyLocation_Cartesian - direction(bb->MyLocation_Cartesian, bb->TargetLocaion_Cartesian, bb->MyForwardVector) * 10000.0 + BT_Geometry::Vector3(0, 0, 300.0);
			mode = THR_MAX;
			break;
		case TASK_MERGE_OFFSET:
		{
			const bool leadTurn = bb->TimeToMerge < number("LeadTurnTTMSec", 1.5);
			// accept_headon (spec 8.4): losing badly in phase 2+ => take the
			// pure trade, skip the lateral offset.
			const bool acceptHeadon = bb->Phase >= 2 &&
				bb->DamageDifference < number("AcceptHeadonIfDiffBelow", -0.30);
			vp = leadTurn ? lead : pure;
			pnAnchor = vp;	// excludes the body-frame offset below on purpose
			if (acceptHeadon)
				vp = alphaBiasedAim(bb, vp, bb->MyRightVector);
			else
			{
				// Even a qualification start inside the gun band keeps the
				// merge offset. Suppressing it converts the opening into the
				// mutual head-on exchange that the BEM maneuver avoids.
				vp = alphaBiasedAim(bb, vp, bb->MyRightVector);
				const double offsetScale = leadTurn ? 0.55 : 1.0;
				vp += turnRight * (side * offsetScale * number("OffsetFt", 600.0) * FT);
			}
			mode = leadTurn ? THR_AUTO : THR_MAX;
			vpTied = true;
			break;
		}
		case TASK_TWO_CIRCLE:
		{
			// Explicit ownship rate-fight turn.  The prior target-tied LagVPP
			// often extended range without accumulating the paper's 180-deg
			// cue.  A lateral ownship-relative VP guarantees horizontal turn
			// authority while holding measured corner speed.
			const double vertical = clampValue(
				bb->TargetLocaion_Cartesian.Z - bb->MyLocation_Cartesian.Z,
				-180.0, 180.0);
			vp = bb->MyLocation_Cartesian +
				turnRight * (side * 5200.0) + turnForward * 900.0 + worldUp * vertical;
			speed = corner;
			mode = bb->MyKCAS_KT < 0.92 * IDENTIFIED_CORNER_KCAS ? THR_MAX : THR_AUTO;
			vpTied = false;
			break;
		}
		case TASK_ONE_CIRCLE:
		{
			// Radius-fight turn: slightly slower than corner speed and a more
			// lateral command point than two-circle.  This reaches the 90-deg
			// cue promptly instead of renaming a lag chase as one-circle.
			const double vertical = clampValue(
				bb->TargetLocaion_Cartesian.Z - bb->MyLocation_Cartesian.Z,
				-160.0, 160.0);
			vp = bb->MyLocation_Cartesian +
				turnRight * (side * 4800.0) + turnForward * 500.0 + worldUp * vertical;
			speed = corner;
			if (bb->MyKCAS_KT > 1.10 * IDENTIFIED_CORNER_KCAS)
				mode = THR_IDLE;
			else if (bb->MyKCAS_KT < 0.96 * IDENTIFIED_CORNER_KCAS)
				mode = THR_MAX;
			else
				mode = THR_AUTO;
			vpTied = false;
			break;
		}
		case TASK_SCISSORS:
		{
			// F-16 BEM 4.5.29: in a flat scissors, pull LV toward the adversary
			// until separation is less than one turn radius, then stop the
			// scissors first by pulling straight up into a stack. Existing
			// altitude separation indicates that the stack already exists.
			const double rangeFt = bb->Distance / FT;
			const bool emergencyConeEscape =
				bb->BFM == DBFM &&
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) > 0.0 &&
				rangeFt < 2600.0;
			if (emergencyConeEscape)
			{
				// Scissors is a DBFM bridge after the bandit's gun solution has
				// been broken.  If the cone is still scoring at close range,
				// continue the immediate cone-escape maneuver first.
				const double defensiveSide = committedDefensiveSide(bb, 1.0);
				vp = enemyConeEscapeVP(bb, defensiveSide, 700.0, 900.0, 9800.0);
				mode = THR_MAX;
				speed = std::max(corner, static_cast<double>(bb->MySpeed_MS) + 15.0);
				break;
			}
			const double altitudeDifference = bb->MyLocation_Cartesian.Z - bb->TargetLocaion_Cartesian.Z;
			const bool dbfmCloseLagOvershoot =
				(bb->BFM == DBFM || bb->LockedManeuverTask == static_cast<int>(TASK_SCISSORS)) &&
				bb->Los_Degree > 110.0f &&
				bb->Los_Degree_Target > 24.0f &&
				rangeFt < 2300.0;
			const double stackEntryRange = dbfmCloseLagOvershoot
				? 2200.0 * FT
				: 1500.0 * FT;
			if (std::abs(altitudeDifference) > 1000.0 * FT)
			{
				const double vertical = altitudeDifference > 0.0 ? 900.0 : -500.0;
				vp = bb->MyLocation_Cartesian + turnRight * (side * 3500.0) +
					turnForward * 1200.0 + worldUp * vertical;
			}
			else if (bb->Distance > stackEntryRange)
				vp = bb->MyLocation_Cartesian + turnRight * (side * 4000.0) +
					turnForward * 800.0;
			else if (dbfmCloseLagOvershoot)
				vp = bb->MyLocation_Cartesian + turnRight * (side * 4200.0) +
					turnForward * 450.0 + worldUp * 850.0;
			else
				vp = bb->MyLocation_Cartesian + turnForward * 600.0 + worldUp * 2800.0;
			speed = std::max(130.0, corner - 50.0 * KNOT);
			if (bb->MyKCAS_KT < 330.0f)
				mode = THR_MAX;
			break;
		}
		case TASK_PURE_PURSUIT:
		{
			const double rangeFt = bb->Distance / FT;
			const double closingFtps = std::max(0.0, -static_cast<double>(bb->ClosureRate_MS) / FT);
			const double speedRatio = bb->MyKCAS_KT > 1.0f
				? static_cast<double>(bb->MyKCAS_KT) / IDENTIFIED_CORNER_KCAS
				: bb->MySpeed_MS / std::max(corner, 1.0);
			if (bb->Los_Degree > 90.0f || rangeFt > 6500.0)
			{
				const double rejoinSide = committedRejoinSide(bb, pure, 1, 5.0);
				const double maxTurnDeg = speedRatio < 0.70 ? 42.0 :
					(speedRatio < 0.90 ? 58.0 : 72.0);
				vp = horizontalRejoinVP(bb, pure, maxTurnDeg, 4800.0, 160.0, rejoinSide);
				mode = THR_MAX;
				speed = std::max(corner, bb->TargetSpeed_MS + 18.0);
				vpTied = false;
			}
			else
			{
				if (rangeFt < phaseMaxRangeFt(bb->Phase) + 2500.0 &&
					bb->Los_Degree < 25.0f &&
					closingFtps > 450.0)
				{
					// Paper OBFM does not continue pure pursuit through a
					// high-closure gun pass. Convert the generic fallback into
					// a control-zone lag command so the following Track phase
					// starts with less closure and more dwell time.
					const double desiredRange = number("HoldRangeFt", 2400.0) * FT;
					const BT_Geometry::Vector3 extendedSix =
						bb->TargetLocaion_Cartesian - bb->TargetForwardVector * desiredRange;
					const BT_Geometry::Vector3 lagVpp = virtualLagVPP(bb, corner, side, 0.70, 1600.0);
					const BT_Geometry::Vector3 rearPoint = extendedSix * 0.45 + lagVpp * 0.55;
					const double pureWeight = clampValue((25.0 - bb->Los_Degree) / 25.0, 0.15, 0.38);
					vp = rearPoint * (1.0 - pureWeight) + pure * pureWeight;
					if (rangeFt < phaseMaxRangeFt(bb->Phase) + 1500.0)
						vp += worldUp * clampValue((closingFtps - 450.0) * 0.35, 0.0, 650.0);
					pnAnchor = vp;
					mode = THR_IDLE;
					speed = std::max(135.0, static_cast<double>(bb->TargetSpeed_MS) - 45.0);
					vpTied = true;
					break;
				}
				vp = pure;
				pnAnchor = pure;
				if (rangeFt > phaseMaxRangeFt(bb->Phase) + 500.0 || bb->Los_Degree > 22.0f)
				{
					mode = (bb->MyKCAS_KT < 0.90 * IDENTIFIED_CORNER_KCAS ||
						rangeFt > phaseMaxRangeFt(bb->Phase) + 1200.0) ? THR_MAX : THR_AUTO;
					speed = std::max(bb->TargetSpeed_MS + 20.0, 0.92 * corner);
				}
				else
					speed = std::max(static_cast<double>(bb->TargetSpeed_MS), 0.84 * corner);
				vpTied = true;
			}
			break;
		}
		case TASK_PULL_TO_HUD:
		{
			const double rangeFt = bb->Distance / FT;
			const double closingFtps = std::max(0.0,
				-static_cast<double>(bb->ClosureRate_MS) / FT);
			Optional<bool> forceLiteralHud = getInput<bool>("ForceLiteralHUD");
			const bool literalHudPull =
				rangeFt < 6500.0 &&
				bb->Los_Degree > 90.0f &&
				((forceLiteralHud && forceLiteralHud.value()) ||
				 bb->BFM == DBFM ||
				 bb->BFM == OBFM ||
				 bb->RunningTime < bb->HABFMPullToHUDUntil);
			if (literalHudPull)
			{
				// Paper block 18 and DBFM lure recovery both say "Pull to HUD".
				// Use the literal pure-HUD pull once the range is bounded and
				// this node is the selected paper bridge.
				vp = pure;
				pnAnchor = pure;
				vpTied = true;
			}
			else if (bb->Los_Degree > 90.0f)
			{
				const double rejoinSide = committedRejoinSide(bb, pure, 1, 3.0);
				vp = horizontalRejoinVP(bb, pure, 65.0, 4800.0, 120.0, rejoinSide);
				vpTied = false;
			}
			else
			{
				const bool energeticLatePhaseControlZonePull =
					bb->Phase >= 2 &&
					bb->BFM == OBFM &&
					bb->ControlZoneState > 0 &&
					rangeFt > 650.0 &&
					rangeFt < phaseMaxRangeFt(bb->Phase) + 800.0 &&
					bb->Los_Degree > 14.0f &&
					bb->Los_Degree < 38.0f &&
					bb->Los_Degree_Target > 45.0f &&
					bb->MyKCAS_KT > 190.0f;
				const bool closeMidAspectScorePull =
					(bb->Phase <= 1 || energeticLatePhaseControlZonePull) &&
					rangeFt > 800.0 &&
					rangeFt < 5800.0 &&
					bb->Los_Degree > 6.0f &&
					bb->Los_Degree < 58.0f &&
					bb->Los_Degree_Target > 6.0f &&
					closingFtps > 450.0 &&
					damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0;
				if (closeMidAspectScorePull)
				{
					// Paper Pull-to-HUD is a terminal weapon-employment bridge,
					// not a neutral pure-pursuit orbit.  In the public cone rule,
					// repeated draws reached 1.5-4.5 kft at 25-45 deg LOS with
					// high closure; pure pursuit then flew through before the nose
					// cone reached the target.  Command a bounded LeadVPP plus a
					// nose-through aim point while keeping the same BT node active.
					const double leadSec = clampValue(0.20 + closingFtps / 5200.0, 0.22, 0.48);
					const double maxLeadDeg = rangeFt < 2400.0 ? 14.0 : 18.0;
					const double leadWeight = clampValue((bb->Los_Degree - 6.0) / 52.0, 0.25, 0.68);
					const BT_Geometry::Vector3 leadAim =
						angularlyLimitedLeadAim(bb, predictedTargetAt(bb, leadSec), maxLeadDeg);
					const BT_Geometry::Vector3 baseAim = pure * (1.0 - leadWeight) + leadAim * leadWeight;
					const double snapGain = rangeFt < 2400.0 ? 1.95 : 1.45;
					const double snapExtraDeg = rangeFt < 2400.0 ? 14.0 : 10.0;
					vp = snapThroughGunAim(bb, baseAim, snapGain, snapExtraDeg);
					pnAnchor = baseAim;
					// Eq. (33): this is the pure-to-lead half of the VPP
					// continuum, so the downstream fixed-gun controller must
					// blend from VPG toward APG instead of reporting no VPP mode.
					bb->VPPMode = VPP_BLEND;
				}
				else
				{
					vp = pure;
					if (bb->Los_Degree < 20.0f)
					{
						// alphaBiasedAim is the velocity-command-side half of the
						// same fixed-gun transition; expose that state to Step().
						vp = alphaBiasedAim(bb, vp, bb->MyRightVector);
						bb->VPPMode = VPP_BLEND;
					}
					else
						bb->VPPMode = VPP_PURE;
					pnAnchor = pure;
				}
				vpTied = true;
			}
			const double desiredRange = number("HoldRangeFt", 1900.0) * FT;
			const double baseSpeed = bb->TargetSpeed_MS +
				0.035 * (bb->Distance - desiredRange) + 0.18 * bb->ClosureRate_MS;
			speed = clampValue(std::max(baseSpeed, 0.74 * corner), 140.0, 280.0);
			const bool preserveTerminalCorner =
				bb->Phase <= 1 &&
				rangeFt > 700.0 &&
				rangeFt < phaseMaxRangeFt(bb->Phase) + 2400.0 &&
				closingFtps > 450.0 &&
				bb->Los_Degree > 12.0f &&
				bb->Los_Degree < 65.0f &&
				(bb->Los_Degree_Target - bb->Los_Degree > 4.0f ||
				 (bb->Los_Degree_Target > 6.0f &&
				  bb->Los_Degree_Target - bb->Los_Degree > -28.0f &&
				  damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0)) &&
				bb->MyKCAS_KT < 0.90 * IDENTIFIED_CORNER_KCAS;
			if (preserveTerminalCorner)
			{
				mode = THR_MAX;
				speed = std::max(speed, std::max(0.98 * corner, static_cast<double>(bb->TargetSpeed_MS) + 20.0));
			}
			else if (rangeFt < number("ClosureBrakeRangeFt", 6400.0) &&
				closingFtps > number("MaxClosureFtps", 300.0))
			{
				mode = THR_IDLE;
				speed = std::max(135.0, static_cast<double>(bb->TargetSpeed_MS) - 38.0);
			}
			else if (rangeFt > 6500.0 || bb->MyKCAS_KT < 0.86 * IDENTIFIED_CORNER_KCAS)
				mode = THR_MAX;
			else
				mode = THR_AUTO;
			break;
		}
		case TASK_ARC_REJOIN:
		{
			const double rangeFt = bb->Distance / FT;
			const double closingFtps = std::max(0.0,
				-static_cast<double>(bb->ClosureRate_MS) / FT);
			const BT_Geometry::Vector3 desired = predictedTargetAt(bb, 0.35);
			const double rejoinSide = committedRejoinSide(bb, desired, 1, 10.0);
			if (bb->Los_Degree > 100.0f)
			{
				vp = horizontalRejoinVP(bb, desired, 92.0, 4400.0, 120.0, rejoinSide);
				vpTied = false;
			}
			else if (bb->Los_Degree > 55.0f)
			{
				const bool farPreScoreRejoin =
					rangeFt > 5200.0 &&
					closingFtps > 250.0 &&
					bb->Los_Degree_Target < 45.0f;
				const BT_Geometry::Vector3 arc = horizontalRejoinVP(
					bb, desired, farPreScoreRejoin ? 104.0 : 72.0,
					farPreScoreRejoin ? 5200.0 : 4300.0, 120.0, rejoinSide);
				const BT_Geometry::Vector3 intercept = angularlyLimitedLeadAim(
					bb, desired, 18.0);
				const double interceptWeight = farPreScoreRejoin
					? clampValue((98.0 - bb->Los_Degree) / 70.0, 0.10, 0.55)
					: clampValue((105.0 - bb->Los_Degree) / 55.0, 0.25, 0.82);
				vp = arc * (1.0 - interceptWeight) + intercept * interceptWeight;
				vpTied = false;
			}
			else
			{
				vp = angularlyLimitedLeadAim(bb, desired, 20.0);
				if (bb->Los_Degree < 18.0f)
					vp = alphaBiasedAim(bb, vp, bb->MyRightVector);
				pnAnchor = desired;
				vpTied = true;
			}
			mode = THR_MAX;
			speed = std::max(corner, bb->TargetSpeed_MS + (rangeFt > 7000.0 ? 28.0 : 18.0));
			break;
		}
		case TASK_NODE35_EXTENDED_SIX:
		{
			// Paper block 35: after a two-circle winning cue, do not jump
			// straight into terminal gun heuristics. Bridge toward the target's
			// extended six o'clock with a lag VPP, then hand off to ControlZone.
			const double holdRangeM = number("HoldRangeFt", 2400.0) * FT;
			const double rangeFt = bb->Distance / FT;
			const BT_Geometry::Vector3 extendedSix =
				bb->TargetLocaion_Cartesian - bb->TargetForwardVector * holdRangeM;
			const BT_Geometry::Vector3 lagVpp = virtualLagVPP(bb, corner, side, 0.65, 1400.0);
			const double lagWeight = clampValue((static_cast<double>(bb->Los_Degree) - 15.0) / 60.0,
				0.25, 0.70);
			vp = extendedSix * (1.0 - lagWeight) + lagVpp * lagWeight;
			bb->VPPMode = lagWeight > 0.55 ? VPP_LAG : VPP_BLEND;
			pnAnchor = vp;
			speed = clampValue(
				std::max(0.90 * corner,
					bb->TargetSpeed_MS + 0.035 * (bb->Distance - holdRangeM) + 0.12 * bb->ClosureRate_MS),
				145.0, 295.0);
			mode = (rangeFt > 5200.0 || bb->MyKCAS_KT < 0.82 * IDENTIFIED_CORNER_KCAS)
				? THR_MAX : THR_AUTO;
			if (rangeFt < 4200.0 && bb->Los_Degree < 45.0f && bb->Los_Degree_Target > 55.0f)
				bb->Node35State = 2;
			bb->OffensiveCommitUntil = static_cast<float>(std::max(
				static_cast<double>(bb->OffensiveCommitUntil), bb->RunningTime + 2.0));
			vpTied = true;
			break;
		}
		case TASK_CONTROL_ZONE:
		{
			const double desiredRange = number("HoldRangeFt", 2300.0) * FT;
			const double rangeFt = bb->Distance / FT;
			const double scoringRangeFt = phaseMaxRangeFt(bb->Phase);
			const double closingFtps = std::max(0.0, -static_cast<double>(bb->ClosureRate_MS) / FT);
			const BT_Geometry::Vector3 extendedSix =
				bb->TargetLocaion_Cartesian - bb->TargetForwardVector * desiredRange;
			const BT_Geometry::Vector3 lagVpp = virtualLagVPP(bb, corner, side, 0.65, 1400.0);
			const bool approach = getInput<bool>("Approach").value_or(false);
			const double lagWeight = approach ? 0.55 : 0.30;
			BT_Geometry::Vector3 rearPoint = extendedSix * (1.0 - lagWeight) + lagVpp * lagWeight;
			double pureWeight = clampValue((85.0 - bb->Los_Degree) / 75.0, 0.0,
				approach ? 0.82 : 0.68) *
				clampValue((1100.0 - closingFtps) / 1100.0, 0.18, 1.0);
			const bool preScoreConeCentering =
				bb->Phase <= 1 &&
				rangeFt > scoringRangeFt - 350.0 &&
				rangeFt < scoringRangeFt + 1300.0 &&
				bb->Los_Degree < 6.0f &&
				bb->Los_Degree_Target > 4.0f &&
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0 &&
				closingFtps > 500.0;
			const bool farPreScoreConeHold =
				bb->Phase <= 1 &&
				rangeFt > scoringRangeFt + 1300.0 &&
				rangeFt < scoringRangeFt + 3300.0 &&
				bb->Los_Degree < 4.5f &&
				bb->Los_Degree_Target > 20.0f &&
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) <= 0.0 &&
				closingFtps > 750.0;
			const bool preScoreSolvedCone =
				preScoreConeCentering &&
				bb->Los_Degree < 2.8f &&
				std::abs(static_cast<double>(bb->MyLosRate_DegSec)) < 6.0;
			const bool enemyActualCone =
				damageRate(bb->Los_Degree_Target, rangeFt, bb->Phase) > 0.0;
			const bool controlZoneTerminalEmploy =
				bb->Phase >= 2 &&
				rangeFt > 700.0 &&
				rangeFt < scoringRangeFt + 900.0 &&
				bb->Los_Degree < 55.0f &&
				bb->Los_Degree_Target > phaseConeDeg(bb->Phase) + 8.0 &&
				closingFtps > 250.0 &&
				!enemyActualCone;
			if (preScoreSolvedCone)
				pureWeight = 1.0;
			else if (farPreScoreConeHold)
				pureWeight = std::max(pureWeight, 0.72);
			else if (preScoreConeCentering)
				pureWeight = std::max(pureWeight, rangeFt <= scoringRangeFt ? 0.94 : 0.72);
			else if (controlZoneTerminalEmploy)
				pureWeight = std::max(pureWeight, 0.92);
			if (rangeFt < 4700.0 && bb->Los_Degree < 58.0f)
				pureWeight = std::max(pureWeight, approach ? 0.48 : 0.38);
			vp = rearPoint * (1.0 - pureWeight) + pure * pureWeight;
			bb->VPPMode = pureWeight > 0.82 ? VPP_PURE : VPP_BLEND;
			if (controlZoneTerminalEmploy)
			{
				// Paper block 12 is a control-zone loop that immediately checks
				// for WEZ/employment.  In the phase-2/3 contest cone, holding a
				// lag-biased control-zone point inside the range band let LOS grow
				// from the 20s to 40+ deg.  Once the approach corridor is already
				// inside the phase range, bias through the nose cone while staying
				// in the same ControlZone task.
				const double snapGain = rangeFt < scoringRangeFt ? 2.25 : 1.75;
				const double snapExtraDeg = rangeFt < scoringRangeFt ? 18.0 : 12.0;
				vp = snapThroughGunAim(bb, pure, snapGain, snapExtraDeg);
				bb->VPPMode = VPP_PURE;
				bb->TrackSubMode = TRACK_SNAP;
			}
			if (preScoreConeCentering && rangeFt <= scoringRangeFt)
			{
				vp = pure;
				bb->VPPMode = VPP_PURE;
			}
			if (preScoreSolvedCone &&
				rangeFt > scoringRangeFt &&
				bb->MyLosRate_DegSec > 2.0f)
			{
				vp = snapThroughGunAim(bb, pure, 1.55, 2.0);
				bb->TrackSubMode = TRACK_SNAP;
			}
			if (!preScoreConeCentering && closingFtps > 500.0 && rangeFt < 4500.0)
				vp += worldUp * clampValue((closingFtps - 500.0) * 0.30, 0.0, 550.0);
			pnAnchor = vp;
			const double commanded = bb->TargetSpeed_MS +
				0.025 * (bb->Distance - desiredRange) + 0.22 * bb->ClosureRate_MS;
			speed = clampValue(std::max(commanded, 0.74 * corner), 135.0, 280.0);
			const bool latePhasePreWezRangeCapture =
				bb->Phase >= 2 &&
				rangeFt > scoringRangeFt &&
				rangeFt < scoringRangeFt + 900.0 &&
				bb->Los_Degree < std::max(12.0, phaseConeDeg(bb->Phase) + 6.0) &&
				bb->Los_Degree_Target > 45.0f &&
				!enemyActualCone;
			if (latePhasePreWezRangeCapture)
			{
				// OBFM flow has not reached "Hostile in WEZ" yet.  Do not
				// bleed closure to hold a nominal zone outside the phase range;
				// capture the range first, then return to the normal zone loop.
				speed = clampValue(std::max(bb->TargetSpeed_MS + 12.0, 0.90 * corner),
					150.0, 330.0);
				mode = THR_MAX;
			}
			else if (closingFtps > 300.0 && rangeFt < 4800.0)
				mode = THR_IDLE;
			else if (rangeFt > 5000.0 || bb->MyKCAS_KT < 0.80 * IDENTIFIED_CORNER_KCAS)
				mode = THR_MAX;
			bb->OffensiveCommitUntil = static_cast<float>(std::max(
				static_cast<double>(bb->OffensiveCommitUntil), bb->RunningTime + 2.0));
			vpTied = true;
			break;
		}
		case TASK_ENERGY_RECOVER:
		{
			// E-M energy management (paper Sec. III; You & Shim eq. 16-19):
			// regain the corner-speed regime. Below corner, trade altitude for
			// speed with a nose-low unload - a climb here would only deepen the
			// stall (observed: a climbing recovery let speed decay to 87 KCAS).
			// At or above corner, bank the surplus into a gentle climb. The hard
			// deck stays guarded by GroundDanger and the VP.z floor clamp below.
			const double vertical = bb->MySpeed_MS < corner ? -1200.0 : 400.0;
			vp = levelForward(bb) * 6000.0 + bb->MyLocation_Cartesian +
				BT_Geometry::Vector3(0, 0, vertical);
			mode = THR_MAX;
			break;
		}
		case TASK_MAINTAIN_MANEUVER:
			break;
		}
		vp.Z = std::max(vp.Z, 2000.0 * FT);

		// Anchor velocity by finite difference. Exact for every anchor motion
		// (lag rotation, near-stationary lead point, turn-circle entry), with
		// task-continuity and jump guards so branch toggles cannot spike PN.
		if (vpTied)
		{
			const double dt = std::max(bb->DeltaSecond, 1e-3);
			bool velocityValid = false;
			if (bb->HasPreviousPNAnchor && bb->PreviousPNTaskKind == static_cast<int>(kind_))
			{
				BT_Geometry::Vector3 anchorVelocity = (pnAnchor - bb->PreviousPNAnchor) / dt;
				if (anchorVelocity.length() < 800.0)
				{
					vpVelocity = anchorVelocity;
					velocityValid = true;
				}
			}
			bb->PreviousPNAnchor = pnAnchor;
			bb->PreviousPNTaskKind = static_cast<int>(kind_);
			bb->HasPreviousPNAnchor = true;
			vpTied = velocityValid;
		}
		else
			bb->HasPreviousPNAnchor = false;

		const BT_Geometry::Vector3 toTarget = bb->TargetLocaion_Cartesian - bb->MyLocation_Cartesian;
		const BT_Geometry::Vector3 toVP = vp - bb->MyLocation_Cartesian;
		bb->NoseToTargetSignedDeg = static_cast<float>(
			signedPlanarAngleDeg(bb->MyForwardVector, toTarget));
		bb->NoseToVPSignedDeg = static_cast<float>(
			signedPlanarAngleDeg(bb->MyForwardVector, toVP));
		bb->SignedTargetErrorDeg = bb->NoseToTargetSignedDeg;
		bb->SignedVPErrorDeg = bb->NoseToVPSignedDeg;
		bb->TargetErrorHorizontalDeg = bb->NoseToTargetSignedDeg;
		bb->TargetErrorVerticalDeg = static_cast<float>(
			signedVerticalErrorDeg(bb->MyForwardVector, bb->MyUpVector, toTarget));
		bb->VPErrorHorizontalDeg = bb->NoseToVPSignedDeg;
		bb->VPErrorVerticalDeg = static_cast<float>(
			signedVerticalErrorDeg(bb->MyForwardVector, bb->MyUpVector, toVP));

		setCommand(bb, vp, mode, speed, vpVelocity, vpTied);
		return NodeStatus::SUCCESS;
	}

	NodeStatus ComputeThrottle::tick()
	{
		CPPBlackBoard* bb = board();
		// thr = trim + Kp*(Vt - V) - Kd*Vdot (spec 6.1); Vdot from the
		// along-track component of the measured acceleration.
		const double alongTrackAccel = bb->MyVelocity.length() > 1.0
			? bb->MyAcceleration.dot(normalized(bb->MyVelocity))
			: 0.0;
		double raw = bb->ThrottleCommandMode == THR_MAX ? 1.0 : (bb->ThrottleCommandMode == THR_IDLE ? 0.0 :
			clampValue(0.65 + number("KpT", 0.02) * (bb->TargetSpeedCommand_MS - bb->MySpeed_MS) -
				number("KdT", 0.01) * alongTrackAccel, 0.0, 1.0));
		const double limit = number("DThrMaxPerTick", 0.05);
		bb->Throttle = static_cast<float>(clampValue(raw, bb->Throttle - limit, bb->Throttle + limit));
		return NodeStatus::SUCCESS;
	}

	void RegisterCompetitionNodes(BehaviorTreeFactory& factory)
	{
#define REGISTER_NODE(Name) factory.registerNodeType<Name>(#Name)
		REGISTER_NODE(UpdateGeometry); REGISTER_NODE(UpdateEnergy); REGISTER_NODE(UpdateEnemyTurnCircle);
		REGISTER_NODE(UpdatePhase); REGISTER_NODE(UpdateDamageScore); REGISTER_NODE(ComputeThrottle);
		REGISTER_NODE(BaselineCorePolicy);
		REGISTER_NODE(DECO_GroundDanger); REGISTER_NODE(DECO_Defensive); REGISTER_NODE(DECO_UnderFire);
		REGISTER_NODE(DECO_PredictedThreat); REGISTER_NODE(DECO_Offensive); REGISTER_NODE(DECO_InMyWEZ);
		REGISTER_NODE(DECO_Overshoot); REGISTER_NODE(DECO_FarBehind); REGISTER_NODE(DECO_EndgameDeny);
		REGISTER_NODE(DECO_FarNeutral); REGISTER_NODE(DECO_PreMerge); REGISTER_NODE(DECO_PostMerge);
		REGISTER_NODE(DECO_EnergyAdvantage); REGISTER_NODE(DECO_Stalemate);
		REGISTER_NODE(DECO_ControlZone); REGISTER_NODE(DECO_MaintainManeuver); REGISTER_NODE(DECO_ThreatCleared);
		REGISTER_NODE(DECO_RoomToManeuver); REGISTER_NODE(DECO_OutsideTurnCircle);
		REGISTER_NODE(DECO_LowEnergy);
		REGISTER_NODE(DECO_CrossingConeLead); REGISTER_NODE(DECO_NearConeAim); REGISTER_NODE(DECO_PendingScissors); REGISTER_NODE(DECO_ShotCommit); REGISTER_NODE(DECO_HABFMBridge); REGISTER_NODE(DECO_ControlZoneCapture);
		REGISTER_NODE(DECO_EnemyPureThreat); REGISTER_NODE(DECO_Node35Active);
		REGISTER_NODE(DECO_EnemyLeadThreat); REGISTER_NODE(DECO_EnemyLagPursuit); REGISTER_NODE(DECO_DefensiveCounterCue);
		REGISTER_NODE(Task_ClimbRecover); REGISTER_NODE(Task_BreakJink); REGISTER_NODE(Task_HardTurn);
		REGISTER_NODE(Task_Track); REGISTER_NODE(Task_HighYoYo); REGISTER_NODE(Task_LeadIntercept);
		REGISTER_NODE(Task_LagEntry); REGISTER_NODE(Task_Extend); REGISTER_NODE(Task_MergeOffset);
		REGISTER_NODE(Task_TwoCircle); REGISTER_NODE(Task_OneCircle); REGISTER_NODE(Task_Scissors);
		REGISTER_NODE(Task_PurePursuit);
		REGISTER_NODE(Task_ControlZone); REGISTER_NODE(Task_MaintainManeuver); REGISTER_NODE(Task_EnergyRecover);
		REGISTER_NODE(Task_FollowHostile);
		REGISTER_NODE(Task_ConeLeadTrack); REGISTER_NODE(Task_PullToHUD); REGISTER_NODE(Task_ArcRejoin);
		REGISTER_NODE(Task_Node35ExtendedSixLag);
#undef REGISTER_NODE
	}
}
