from abc import ABC
from random import random, uniform, choice
from typing import List, Union, Dict
from unittest.mock import MagicMock

from .obstacle import Obstacle
from ..game_entities.entity import GameEntity


class Traveller(ABC):
    """Application mixin class to support moving around the map."""

    COURSE: List[Obstacle] = []

    LONG_OBSTACLE_TIMEOUT = 60
    MED_OBSTACLE_TIMEOUT = 6
    SHORT_OBSTACLE_TIMEOUT = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # these attributes will be initialised in the mixin using the
        # traveller_init method
        self.obstacle_id = None
        self.path: Union[List, None] = None
        self.started_course: bool = False
        self.course_loop: bool = False

        # these attributes won't be initialised since this is a mixin class,
        # but they are here for documentation / linting purposes
        self.entities: List[GameEntity] = []
        self.entities_mapping: Dict[str, str] = {}
        self.client = None
        self.afk_timer = MagicMock()
        self.msg = []
        self._setup_game_entity = lambda *_, **__: MagicMock()
        self._teleport_with_item = lambda *_, **__: False
        self.swap_confidence: Union[float, None] = None

    def traveller_init(self):
        self.obstacle_id: int = 0
        self.path: Union[List, None] = []
        self.swap_confidence: float = -float('inf')
        self.started_course = False
        self.course_loop: bool = False

    @property
    def current_obstacle(self) -> Obstacle:
        """Get the next obstacle we need to cross."""
        return self.COURSE[self.obstacle_id]

    @property
    def next_obstacle(self) -> Obstacle:
        """Get the next obstacle we need to cross.

        If the course is a loop"""
        try:
            return self.COURSE[self.obstacle_id + 1]
        except IndexError:
            if self.course_loop:
                return self.COURSE[0]
            return self.COURSE[-1]

    def set_started_course(self):
        """Set the started course flag."""
        self.started_course = True

    def unset_started_course(self):
        """Unset the started course flag."""
        self.started_course = False

    def obstacle_post_script(self):
        """
        set a small afk timer after we pass an obstacle, so we don't
        appear to always do an immediate action.

        Also update entities now that the map has been swapped entities need
        to be re-updated to pick up their new locations.
        """

        # some obstacles require an extra delay after arriving, usually because
        # we actually "arrive" too early, e.g. if the arrival node is the other
        # end of a pipe on the same map with range 2 - we "arrive" on the wrong
        # side and still need a few ticks to get to the other side
        self.afk_timer.add_timeout(
            self.client.TICK * self.current_obstacle.additional_delay
        )

        afk = random()
        if afk < 0.0001:
            self.afk_timer.add_timeout(uniform(
                self.client.TICK,
                self.client.TICK * self.LONG_OBSTACLE_TIMEOUT
            ))
        elif afk < 0.01:
            self.afk_timer.add_timeout(uniform(
                self.client.TICK,
                self.client.TICK * self.MED_OBSTACLE_TIMEOUT
            ))
        elif afk < 0.1:
            self.afk_timer.add_timeout(uniform(
                self.client.TICK,
                self.client.TICK * self.SHORT_OBSTACLE_TIMEOUT
            ))

        self.client.logger.debug(f'Calculated afk to: {afk:.4f}')

    def increment_obstacle_id(self):
        """Increment obstacle id by one."""
        self.obstacle_post_script()
        self.obstacle_id = self.obstacle_id + 1

    def reset_course(self):
        """Clear the timeout for the all obstacles and start at index 0."""
        self.unset_started_course()
        for obstacle in self.COURSE:
            obstacle.entity.clear_timeout()
        self.obstacle_id = 0

    def remove_course(self):
        """Remove all obstacles from the course, and remove entities from
        entities list."""
        for obstacle in self.COURSE:
            entity = obstacle.entity
            idx = self.entities.index(entity)
            self.entities.pop(idx)
        self.COURSE = []

    def setup_course(self):
        """Create entities for all obstacles in the course."""

        gps = self.client.gauges.minimap.gps

        for obstacle in self.COURSE:
            map_ = gps.load_map(obstacle.map_name, set_current=False)
            # TODO: NPCs as obstacles.

            # skip if we've already created an entity for this obstacle
            if obstacle.entity is not None:
                continue

            entity = self._setup_game_entity(
                obstacle.label, map_=map_
            )
            if obstacle.offsets:
                entity.x1_offset = obstacle.offsets[0]
                entity.y1_offset = obstacle.offsets[1]
                entity.x2_offset = obstacle.offsets[2]
                entity.y2_offset = obstacle.offsets[3]

            # TODO: custom clickbox?

            # cache entity to obstacle item and application class for updating
            obstacle.entity = entity
            self.entities.append(entity)

            # add mapping for entity to map name, so we can update the entity
            # only when we're on that map.
            self.entities_mapping[obstacle.entity.id] = obstacle.map_name

    def update_path(self):

        mm = self.client.gauges.minimap
        gps = self.client.gauges.minimap.gps
        pxy = gps.get_coordinates()
        px, py = pxy

        # trim nodes in path we're close enough to or already passed
        trim_idx = 0
        for i, entity in enumerate(self.path):
            node = entity.get_global_coordinates()
            dist = mm.distance_between(node, pxy)
            if dist > (mm.orb_radius - 10) / mm.tile_size:
                break

            trim_idx = i
        self.path = self.path[trim_idx:]

        # check the first node in the path is actually reachable
        for entity in self.path:
            node = entity.get_global_coordinates()
            dist = mm.distance_between(node, (px, py), as_pixels=True)
            if dist > mm.orb_radius:
                # reset the path to recalculate
                self.path = []
            break

        # update each node in path with updated keys
        for entity in self.path:
            x, y = entity.get_global_coordinates()
            key = (int((x - px) * mm.tile_size),
                   int((y - py) * mm.tile_size))
            entity.update(key=key)

    def travel_path(self, speed):

        mm = self.client.gauges.minimap
        gps = self.client.gauges.minimap.gps

        if self.path:
            checkpoint = self.path[0]
            node = checkpoint.get_global_coordinates()
            dist = mm.distance_between(node, gps.get_coordinates())
            if not speed or not checkpoint.clicked:

                # expand the minimap bbox by 2 tiles
                x1, y1, x2, y2 = checkpoint.mm_bbox()
                x1 -= mm.tile_size * 2
                y1 -= mm.tile_size * 2
                x2 += mm.tile_size * 2
                y2 += mm.tile_size * 2

                checkpoint.click(
                    # divide 2 assumes running
                    dist * self.client.TICK / 2,
                    dist * self.client.TICK / 2 * (random() + 1),
                    bbox=(x1, y1, x2, y2),
                    pause_before_click=True,
                    speed=0.1,
                )
                self.afk_timer.add_timeout(uniform(.2, .6))
                self.msg.append(f'clicked checkpoint {node}')
            else:
                self.msg.append(f'waiting arrive checkpoint: {node}')
        else:
            start = gps.current_map.find(nearest=gps.get_coordinates())
            end = gps.current_map.label_to_node(
                self.current_obstacle.label).pop()
            end = gps.current_map.find(nearest=end)

            # ensure path is a clean list
            self.path = []

            if self.current_obstacle.routes:
                checkpoint = [choice(self.current_obstacle.routes)]
            else:
                checkpoint = None

            try:
                path = gps.get_route(start, end, checkpoints=checkpoint)
            except KeyError as err:
                self.msg.append(f'failed to generate path: {err}')
                return

            # if self.current_obstacle.success_label == 'cave_3_exit':
            #     self.client.logger.debug('trimmed last node')
            #     path = path[:-1]

            px, py = gps.get_coordinates()
            for x, y in path:
                key = (int((x - px) * mm.tile_size),
                       int((y - py) * mm.tile_size))

                entity = self.client.game_screen.create_game_entity(
                    'node', 'node', key, self.client, self.client
                )
                entity.set_global_coordinates(x, y)
                self.path.append(entity)

            self.msg.append(f'generated path ({speed:.1f}) from: {start} to {end}')

    def travel_course(self):

        mm = self.client.gauges.minimap
        gps = self.client.gauges.minimap.gps
        speed = gps.calculate_average_speed(period=self.client.TICK)

        x1, y1, x2, y2 = self.current_obstacle.entity.click_box()
        if self.client.game_screen.is_clickable(
                x1, y1, x2, y2,
                allow_partial=self.current_obstacle.allow_partial):

            dist = mm.distance_between(
                gps.get_coordinates(),
                self.current_obstacle.entity.get_global_coordinates()
            )
            # TODO: check if running
            dist_timeout = max([dist * (self.client.TICK / 2),
                                self.client.TICK])
            timeout = dist_timeout + self.current_obstacle.timeout

            method = None
            if self.current_obstacle.custom_click_box:
                method = self.current_obstacle.entity.click_box

            self.path = []  # clear path since obstacle is on screen

            # if map swap confidence is so low at current obstacle it's easy to
            # get stuck. However, the previous map confidence is also low,
            # so we can use that to make an assessment.
            before = self.current_obstacle.fallback_confidence_before
            after = self.current_obstacle.fallback_confidence_after
            map_swap_fallback = (
                    gps.confidence <= before
                    and self.swap_confidence <= after
            )

            if map_swap_fallback:
                gps.load_map(self.next_obstacle.map_name)
                node = gps.current_map.label_to_node(
                    self.current_obstacle.success_label).pop()
                gps.set_coordinates(*node)
                self.msg.append(
                    f'Obstacle fallback skip: '
                    f'{self.current_obstacle.label}')
                self.increment_obstacle_id()

            self._teleport_with_item(
                self.current_obstacle.entity,
                self.next_obstacle.map_name,
                self.current_obstacle.success_label,
                tmin=timeout,
                tmax=timeout * 1.5,
                post_script=self.increment_obstacle_id,
                mouse_text=self.current_obstacle.mouse_text,
                multi=self.current_obstacle.multi,
                confidence=self.current_obstacle.min_confidence,
                method=method,
                range_=self.current_obstacle.range,
            )
        elif not self.current_obstacle.entity.clicked:
            self.travel_path(speed)
        else:
            self.msg.append(
                f'waiting arrive obstacle: '
                f'{self.current_obstacle.entity.time_left:.3f}'
            )
